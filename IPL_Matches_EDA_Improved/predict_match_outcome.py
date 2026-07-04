from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "IPL_Matches_2008_2022.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    matches = pd.read_csv(path)
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    matches["Season"] = matches["Date"].dt.year

    rename_map = {
        "Kings XI Punjab": "Punjab Kings",
        "Delhi Daredevils": "Delhi Capitals",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
    }
    for col in ["Team1", "Team2", "WinningTeam", "TossWinner"]:
        matches[col] = matches[col].replace(rename_map)

    matches["WinningTeam"] = matches["WinningTeam"].fillna("NoResult")
    matches["TossWinner"] = matches["TossWinner"].fillna("NoResult")
    matches["Team1Wins"] = (matches["WinningTeam"] == matches["Team1"]).astype(int)
    matches["TossWinnerIsTeam1"] = (matches["TossWinner"] == matches["Team1"]).astype(int)

    return matches


def build_model(matches: pd.DataFrame):
    model_df = matches[["Team1", "Team2", "Venue", "TossDecision", "TossWinnerIsTeam1", "Team1Wins"]].dropna().copy()

    X = model_df[["Team1", "Team2", "Venue", "TossDecision", "TossWinnerIsTeam1"]]
    y = model_df["Team1Wins"]

    categorical_features = ["Team1", "Team2", "Venue", "TossDecision"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("passthrough", "passthrough", ["TossWinnerIsTeam1"]),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    return pipeline, X_test, y_test, predictions, accuracy, report, cm


def create_plots(matches: pd.DataFrame):
    team_counts = pd.concat([matches["Team1"], matches["Team2"]]).value_counts().reset_index()
    team_counts.columns = ["Team", "MatchesPlayed"]
    win_counts = matches["WinningTeam"].value_counts().reset_index()
    win_counts.columns = ["Team", "Wins"]
    summary = team_counts.merge(win_counts, on="Team", how="left").fillna(0)
    summary["WinRatio"] = (summary["Wins"] / summary["MatchesPlayed"] * 100).round(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=summary.sort_values("MatchesPlayed", ascending=False).head(10),
        x="MatchesPlayed",
        y="Team",
        hue="Team",
        dodge=False,
        legend=False,
        palette="viridis",
        ax=axes[0],
    )
    axes[0].set_title("Most Active IPL Teams")
    axes[0].set_xlabel("Matches Played")
    axes[0].set_ylabel("Team")

    sns.barplot(
        data=summary.sort_values("WinRatio", ascending=False).head(10),
        x="WinRatio",
        y="Team",
        hue="Team",
        dodge=False,
        legend=False,
        palette="magma",
        ax=axes[1],
    )
    axes[1].set_title("Top Win Ratio by Team")
    axes[1].set_xlabel("Win Percentage")
    axes[1].set_ylabel("Team")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "team_summary.png", dpi=300)
    plt.close()

    toss_decision = matches["TossDecision"].value_counts().reset_index()
    toss_decision.columns = ["Decision", "Count"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=toss_decision, x="Decision", y="Count", hue="Decision", dodge=False, legend=False, palette="Set2", ax=ax)
    ax.set_title("Toss Decision Distribution")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "toss_decision.png", dpi=300)
    plt.close()


def save_metrics(accuracy: float, report: dict, cm, output_path: Path):
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def main():
    matches = load_and_prepare_data(DATA_PATH)
    create_plots(matches)
    _, _, _, _, accuracy, report, cm = build_model(matches)
    save_metrics(accuracy, report, cm, OUTPUT_DIR / "model_metrics.json")

    print("Dataset shape:", matches.shape)
    print("Model accuracy:", round(accuracy, 4))
    print("Saved plots and metrics to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
