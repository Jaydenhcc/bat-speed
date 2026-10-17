"""
swing_gain.py

Swing Gain (SG): does swinging harder lead to better run value for a hitter?

For each hitter we:
  1. Keep only swings (whiffs, fouls, balls in play).
  2. Drop the slowest 10% of that hitter's swings (likely defensive "sword" swings).
  3. Fit a straight line: bat speed (x) vs. delta_run_exp (y).
  4. Convert the slope to Swing Gain = slope * 2 mph * 100 swings,
     with a 95% confidence interval.

Bat speed is only tracked by Statcast from 2023 onward.

Use it from a notebook:

    import sys
    sys.path.append("../src")
    import swing_gain as sg
    df = sg.load_statcast(2025)
    results = sg.analyze_team(df, "NYY")

Or from the terminal (run from the project folder):

    python src/swing_gain.py --season 2025 --team NYY
    python src/swing_gain.py --season 2025 --player "Aaron Judge"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import linregress, t
from pybaseball import playerid_lookup, playerid_reverse_lookup, statcast

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs"

# Pitch outcomes that count as a swing (takes like "ball" are left out).
SWING_DESCRIPTIONS = [
    "hit_into_play",
    "foul",
    "foul_tip",
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "foul_bunt",
]

OUTCOME_COLORS = {
    "hit_into_play": "steelblue",
    "foul": "gold",
    "foul_tip": "gold",
    "foul_bunt": "gold",
    "swinging_strike": "crimson",
    "swinging_strike_blocked": "crimson",
    "missed_bunt": "crimson",
}

RECOMMENDATION_COLORS = {
    "swing harder": "seagreen",
    "swing easier": "crimson",
    "no clear signal": "gray",
}

SUMMARY_COLS = ["player_name", "SG", "SG_low", "SG_high", "n_swings", "recommendation"]
DETAIL_COLS = [
    "player_name", "slope", "intercept", "r_squared", "p_value",
    "mean_bat_speed", "mean_delta_run_exp", "sword_speed_cutoff", "n_swords_excluded",
]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_statcast(season, raw_dir=RAW_DIR, force_refresh=False):
    """
    Return regular season Statcast pitches for one season.

    The first call downloads the data (slow, can take many minutes) and saves it
    to data/raw/statcast_<season>.parquet. Later calls read that file instead.
    Set force_refresh=True to download again.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_file = raw_dir / f"statcast_{season}.parquet"

    if cache_file.exists() and not force_refresh:
        df = pd.read_parquet(cache_file)
    else:
        df = statcast(start_dt=f"{season}-03-01", end_dt=f"{season}-11-01")
        df.to_parquet(cache_file)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df[df["game_type"] == "R"].copy()


# ---------------------------------------------------------------------------
# 2. Player lookup
# ---------------------------------------------------------------------------

def get_player_id(first, last):
    """Look up a player's MLBAM ID from a first and last name."""
    matches = playerid_lookup(last, first).dropna(subset=["key_mlbam"])
    if matches.empty:
        raise LookupError(f"No MLBAM player ID found for {first} {last}.")
    if len(matches) > 1:
        raise LookupError(
            f"{len(matches)} players matched {first} {last}. "
            "Pass the key_mlbam ID directly instead."
        )
    return int(matches.iloc[0]["key_mlbam"])


def get_player_name(player_id):
    """Turn an MLBAM ID back into a readable name."""
    lookup = playerid_reverse_lookup([int(player_id)])
    if lookup.empty:
        return f"Unknown ({player_id})"
    row = lookup.iloc[0]
    return f"{row['name_first']} {row['name_last']}".title()


# ---------------------------------------------------------------------------
# 3. Swing cleaning
# ---------------------------------------------------------------------------

def remove_sword_swings(swings, percentile=0.10):
    """
    Drop a hitter's slowest swings, which are likely defensive swings.

    Returns the kept swings, the speed cutoff, and the number removed.
    Rows with no bat speed are left alone.
    """
    speeds = pd.to_numeric(swings["bat_speed"], errors="coerce")
    cutoff = speeds.quantile(percentile)
    is_sword = speeds < cutoff
    kept = swings.loc[~is_sword].copy()
    return kept, cutoff, int(is_sword.sum())


def get_player_swings(statcast_df, player_id):
    """
    Get one hitter's clean swings, ready for the regression.

    Returns (swings, sword_cutoff, n_swords_excluded), or None if the hitter
    has no bat speed data. Every row in `swings` has both bat_speed and
    delta_run_exp.
    """
    player_df = statcast_df.loc[statcast_df["batter"].eq(int(player_id))]
    swings = player_df.loc[player_df["description"].isin(SWING_DESCRIPTIONS)].copy()

    swings["bat_speed"] = pd.to_numeric(swings["bat_speed"], errors="coerce")
    swings["delta_run_exp"] = pd.to_numeric(swings["delta_run_exp"], errors="coerce")
    swings = swings.dropna(subset=["bat_speed"])

    if swings.empty:
        return None

    kept, cutoff, n_swords = remove_sword_swings(swings)
    kept = kept.dropna(subset=["delta_run_exp"])
    return kept, cutoff, n_swords


# ---------------------------------------------------------------------------
# 4. The metric
# ---------------------------------------------------------------------------

def swing_gain(result, n_swings, mph_change=2, per_swing=100, confidence=0.95):
    """
    Turn a linregress result into Swing Gain (SG).

    SG = expected change in runs per 100 swings if the hitter swings
    mph_change mph harder. Also returns the confidence interval and a label.
    """
    scale = mph_change * per_swing
    sg = result.slope * scale

    degrees_of_freedom = n_swings - 2
    t_critical = t.ppf(1 - (1 - confidence) / 2, degrees_of_freedom)
    margin = t_critical * result.stderr * scale
    sg_low = sg - margin
    sg_high = sg + margin

    if sg_low > 0:
        recommendation = "swing harder"
    elif sg_high < 0:
        recommendation = "swing easier"
    else:
        recommendation = "no clear signal"

    return {
        "SG": sg,
        "SG_low": sg_low,
        "SG_high": sg_high,
        "recommendation": recommendation,
    }


def analyze_swing_decision(statcast_df, player_id, min_swings=100):
    """
    Run the full analysis for one batter.

    Returns a dict of results, or None if the batter has fewer than
    min_swings usable swings.
    """
    player_id = int(player_id)
    cleaned = get_player_swings(statcast_df, player_id)
    if cleaned is None:
        return None

    swing_rv, sword_cutoff, n_swords_excluded = cleaned
    if len(swing_rv) < min_swings:
        return None

    x = swing_rv["bat_speed"]
    y = swing_rv["delta_run_exp"]
    result = linregress(x, y)

    return {
        "player_id": player_id,
        "player_name": get_player_name(player_id),
        **swing_gain(result, n_swings=len(swing_rv)),
        "n_swings": len(swing_rv),
        "n_swords_excluded": n_swords_excluded,
        "sword_speed_cutoff": sword_cutoff,
        "mean_bat_speed": x.mean(),
        "mean_delta_run_exp": y.mean(),
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.rvalue ** 2,
        "p_value": result.pvalue,
    }


# ---------------------------------------------------------------------------
# 5. Many hitters at once
# ---------------------------------------------------------------------------

def team_batter_ids(statcast_df, team):
    """All batter IDs who batted for a team (use the abbreviation, like "NYY")."""
    inning_half = statcast_df["inning_topbot"].astype("string").str.lower()
    batting_mask = (
        (statcast_df["home_team"].eq(team) & inning_half.eq("bot"))
        | (statcast_df["away_team"].eq(team) & inning_half.eq("top"))
    )
    ids = pd.to_numeric(statcast_df.loc[batting_mask, "batter"], errors="coerce")
    return sorted(ids.dropna().astype(int).unique())


def analyze_batters(statcast_df, batter_ids, min_swings=100):
    """Analyze a list of batter IDs. Returns a DataFrame sorted by SG, best first."""
    rows = []
    for batter_id in batter_ids:
        row = analyze_swing_decision(statcast_df, batter_id, min_swings=min_swings)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["player_id"] + DETAIL_COLS + SUMMARY_COLS)
    return pd.DataFrame(rows).sort_values("SG", ascending=False).reset_index(drop=True)


def analyze_team(statcast_df, team, min_swings=100):
    """Analyze every hitter on one team."""
    return analyze_batters(statcast_df, team_batter_ids(statcast_df, team), min_swings)


def analyze_league(statcast_df, min_swings=100):
    """
    Analyze every hitter in the data.

    This is slow because each hitter needs a name lookup.
    """
    all_ids = sorted(
        pd.to_numeric(statcast_df["batter"], errors="coerce").dropna().astype(int).unique()
    )
    return analyze_batters(statcast_df, all_ids, min_swings)


def summary_table(results_df):
    """The short leaderboard: SG, interval, swing count, and label."""
    return results_df[SUMMARY_COLS].round(2)


def details_table(results_df):
    """The regression details: slope, intercept, R squared, p value, and more."""
    return results_df[DETAIL_COLS].round(4)


# ---------------------------------------------------------------------------
# 6. Plots (each returns a figure so you can show it or save it)
# ---------------------------------------------------------------------------

def plot_bat_speed_hist(swing_rv, player_name):
    """Histogram of a hitter's bat speed, with the average and 90th percentile."""
    speeds = swing_rv["bat_speed"]
    p90 = speeds.quantile(0.90)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(speeds, bins=15, color="steelblue", edgecolor="white")
    ax.axvline(speeds.mean(), color="darkorange", linestyle="--", linewidth=2,
               label=f"Average: {speeds.mean():.1f} mph")
    ax.axvline(p90, color="crimson", linestyle="--", linewidth=2,
               label=f"90th percentile: {p90:.1f} mph")
    ax.set_title(f"{player_name} bat speed (slowest 10% removed)")
    ax.set_xlabel("Bat speed (mph)")
    ax.set_ylabel("Number of swings")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bat_speed_vs_run_value(swing_rv, player_name, season=None, n_bins=8):
    """Scatter of every swing, group averages with 95% CI, and the linear fit."""
    fit = linregress(swing_rv["bat_speed"], swing_rv["delta_run_exp"])
    gain = swing_gain(fit, n_swings=len(swing_rv))
    colors = swing_rv["description"].map(OUTCOME_COLORS).fillna("gray")

    binned = swing_rv.copy()
    binned["speed_bin"] = pd.qcut(binned["bat_speed"], q=n_bins, duplicates="drop")
    bin_stats = binned.groupby("speed_bin", observed=True).agg(
        mean_speed=("bat_speed", "mean"),
        mean_rv=("delta_run_exp", "mean"),
        sem_rv=("delta_run_exp", "sem"),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(swing_rv["bat_speed"], swing_rv["delta_run_exp"], c=colors, alpha=0.3, s=20)
    ax.errorbar(bin_stats["mean_speed"], bin_stats["mean_rv"], yerr=1.96 * bin_stats["sem_rv"],
                fmt="o", color="black", capsize=4, markersize=7, label="Group average (95% CI)")

    x_fit = pd.Series(
        [swing_rv["bat_speed"].min(), swing_rv["bat_speed"].max()]
    )
    ax.plot(x_fit, fit.slope * x_fit + fit.intercept, color="black",
            linewidth=2, linestyle="--", label="Linear fit")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)

    outcome_handles = [
        Patch(facecolor="steelblue", label="Hit into play"),
        Patch(facecolor="gold", label="Foul"),
        Patch(facecolor="crimson", label="Swinging strike"),
    ]
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=outcome_handles + line_handles)

    season_text = f", {season} regular season" if season else ""
    ax.set_title(
        f"{player_name}{season_text}: bat speed vs. run value\n"
        f"Swing Gain {gain['SG']:+.2f} runs per 100 swings "
        f"(95% CI {gain['SG_low']:+.2f} to {gain['SG_high']:+.2f}): {gain['recommendation']}"
    )
    ax.set_xlabel("Bat speed (mph)")
    ax.set_ylabel("Change in run expectancy")
    fig.tight_layout()
    return fig


def plot_leaderboard(results_df, title="Swing Gain leaderboard"):
    """Horizontal chart of SG with 95% intervals, colored by recommendation."""
    df = results_df.sort_values("SG", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(df) + 1.5)))
    for i, row in df.iterrows():
        color = RECOMMENDATION_COLORS[row["recommendation"]]
        ax.errorbar(row["SG"], i,
                    xerr=[[row["SG"] - row["SG_low"]], [row["SG_high"] - row["SG"]]],
                    fmt="o", color=color, capsize=3, markersize=7)

    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["player_name"])
    ax.set_xlabel("Swing Gain (runs per 100 swings for +2 mph), with 95% interval")
    ax.set_title(title)
    ax.legend(handles=[Patch(facecolor=c, label=label)
                       for label, c in RECOMMENDATION_COLORS.items()],
              loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Command line use
# ---------------------------------------------------------------------------

def _run_player(df, full_name, season, min_swings, out_dir):
    first, _, last = full_name.strip().partition(" ")
    player_id = get_player_id(first, last)
    name = get_player_name(player_id)

    cleaned = get_player_swings(df, player_id)
    if cleaned is None:
        print(f"No bat speed data found for {name}.")
        return
    swing_rv, _, _ = cleaned

    row = analyze_swing_decision(df, player_id, min_swings=min_swings)
    if row is None:
        print(f"{name} has fewer than {min_swings} usable swings.")
        return

    print(pd.Series({k: row[k] for k in SUMMARY_COLS}).to_string())

    slug = name.lower().replace(" ", "_")
    plot_bat_speed_hist(swing_rv, name).savefig(out_dir / f"{slug}_{season}_hist.png", dpi=150)
    plot_bat_speed_vs_run_value(swing_rv, name, season).savefig(
        out_dir / f"{slug}_{season}_scatter.png", dpi=150)
    print(f"Saved charts to {out_dir}")


def _run_team(df, team, season, min_swings, out_dir):
    results = analyze_team(df, team, min_swings=min_swings)
    if results.empty:
        print(f"No hitters met the {min_swings} swing minimum for {team}.")
        return

    print(summary_table(results).to_string())
    summary_table(results).to_csv(out_dir / f"{team}_{season}_summary.csv", index=False)
    details_table(results).to_csv(out_dir / f"{team}_{season}_details.csv", index=False)
    plot_leaderboard(results, f"{team} {season} Swing Gain").savefig(
        out_dir / f"{team}_{season}_leaderboard.png", dpi=150)
    print(f"Saved tables and chart to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Swing Gain: bat speed vs. run value.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--team", help='Team abbreviation, like "NYY"')
    parser.add_argument("--player", help='Full name, like "Aaron Judge"')
    parser.add_argument("--min_swings", type=int, default=100)
    parser.add_argument("--raw_dir", default=str(RAW_DIR))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--refresh", action="store_true",
                        help="Download the season again instead of using the saved file")
    args = parser.parse_args()

    if not args.team and not args.player:
        parser.error("Give me --team or --player.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_statcast(args.season, raw_dir=args.raw_dir, force_refresh=args.refresh)

    if args.player:
        _run_player(df, args.player, args.season, args.min_swings, out_dir)
    if args.team:
        _run_team(df, args.team.upper(), args.season, args.min_swings, out_dir)


if __name__ == "__main__":
    main()
