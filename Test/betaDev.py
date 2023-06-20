import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns

sns.set_theme()


class UserHistory:
    def __init__(self, src: list, id=None):
        if id:
            self.id = id
        else:
            self.id = input("Enter Username: ")
        self.data = self.__prep_user_data(src)

    def __prep_user_data(self, src):
        if len(src) < 1:
            return

        df = pd.read_json(src[0])
        if len(src) > 1:
            for index, file in enumerate(src, start=1):
                df = pd.concat([df, pd.read_json(file)], ignore_index=True)

        data = df
        data["endTime"] = pd.to_datetime(data["endTime"], format="%Y-%m-%d %H:%M")
        data["day"] = data["endTime"].dt.strftime("%d")
        data["hour"] = data["endTime"].dt.strftime("%H")
        data["month"] = data["endTime"].dt.strftime("%m")
        data["dm"] = data["endTime"].dt.strftime("%m/%d/%Y")

        def get_season(date):
            d = date.split("/")
            month = int(d[0])
            day = int(d[1])
            if month >= 3 and month < 6:
                return "spring"
            elif month >= 6 and month < 9:
                return "summer"
            elif month >= 9 and month < 12:
                return "fall"
            else:
                return "winter"

        data["season"] = data["dm"].apply(get_season)
        return data

    def run(self):
        print("\n")
        print(f"-------Starting Analytics for {self.id}------- \n")
        print(f"-------{self.id}'s top 10 artists------- \n")
        print(self.get_user_data("artistName"))
        print("\n")
        print(f"-------{self.id}'s top 10 songs------- \n")
        print(self.get_user_data("trackName"))
        print("\n")
        print(f"-------{self.id}'s Seasonal Breakdown------- \n")
        print(f"-------Winter------- \n")
        print(self.get_season_son("winter"))
        print(self.get_season_art("winter"))
        print("\n")
        print(f"-------{self.id}'s Seasonal Breakdown------- \n")
        print(self.get_hours())

    def get_user_data(self, context: str, ranks=10):
        df = self.data.copy()
        grouped_df = df.groupby(context)["msPlayed"].sum().reset_index()
        grouped_df["rank"] = grouped_df["msPlayed"].rank(
            method="average", ascending=False
        )
        grouped_df["minutesPlayed"] = round((grouped_df["msPlayed"] / 60000), 2)
        sorted_df = grouped_df.sort_values("msPlayed", ascending=False)
        sorted_df.drop(["msPlayed"], axis=1, inplace=True)
        return sorted_df[sorted_df["rank"] <= ranks]

    def get_artists(self, data):
        df = data.copy()
        grouped_df = df.groupby("artistName")["msPlayed"].sum().reset_index()
        grouped_df["rank"] = grouped_df["msPlayed"].rank(
            method="average", ascending=False
        )
        grouped_df["minutesPlayed"] = round((grouped_df["msPlayed"] / 60000), 2)
        sorted_df = grouped_df.sort_values("msPlayed", ascending=False)
        sorted_df.drop(["msPlayed"], axis=1, inplace=True)
        return sorted_df[sorted_df["rank"] <= 10]

    def get_songs(self, data):
        df = data.copy()
        grouped_df = df.groupby("trackName")["msPlayed"].sum().reset_index()
        grouped_df["rank"] = grouped_df["msPlayed"].rank(
            method="average", ascending=False
        )
        grouped_df["minutesPlayed"] = round((grouped_df["msPlayed"] / 60000), 2)
        sorted_df = grouped_df.sort_values("msPlayed", ascending=False)
        sorted_df.drop(["msPlayed"], axis=1, inplace=True)
        return sorted_df[sorted_df["rank"] <= 10]

    def get_month_art(self, filter_month):
        return self.get_artists(self.data[self.data["month"] == filter_month])

    def get_month_son(self, filter_month):
        return self.get_songs(self.data[self.data["month"] == filter_month])

    def get_top_mont(self, data):
        df = data.copy()
        grouped_df = df.groupby("month")["msPlayed"].sum().reset_index()
        sorted_df = grouped_df.sort_values("msPlayed", ascending=False)
        sorted_df["minutesPlayed"] = sorted_df["msPlayed"] / 60000
        sorted_df.drop(["msPlayed"], axis=1, inplace=True)
        return sorted_df

    def month_breakdown(self, filter_month, plot=False):
        data = self.data
        df = data[data["month"] == filter_month]
        grouped_df = df.groupby("day")["msPlayed"].sum().reset_index()
        sorted_df = grouped_df.sort_values("msPlayed", ascending=False)
        sorted_df["minutesPlayed"] = sorted_df["msPlayed"] / 60000
        sorted_df.drop(["msPlayed"], axis=1, inplace=True)
        if plot:
            self.display_month(sorted_df, filter_month)
        return sorted_df

    def get_season_art(self, season):
        return self.get_artists(self.data[self.data["season"] == season])

    def get_season_son(self, season):
        return self.get_songs(self.data[self.data["season"] == season])

    def display_hour(self, curDf):
        ax = curDf.plot.bar(
            x="hour",
            y="AvgMinutesPlayed",
            rot=0,
            figsize=(10, 10),
            title=f"Average Daily Streaming for User",
            xlabel="Hour",
            ylabel="Minutes Spent",
            fontsize=12,
            color="red",
        )
        ax.bar_label(ax.containers[0], fmt="%.1f")
        plt.show()

    def display_month(self, curDf, mon):
        ax = curDf.plot.bar(
            x="day",
            y="minutesPlayed",
            rot=0,
            figsize=(30, 10),
            title=f"Daily User Breakdown for {mon}",
            xlabel="Hour",
            ylabel="Average Minutes Spent",
            fontsize=12,
            color="blue",
        )
        ax.bar_label(ax.containers[0], fmt="%.1f")
        plt.show()

    def get_hours(self, plot=False):
        df = self.data.copy()
        sorted_df = df.groupby("hour")["msPlayed"].mean().reset_index()

        sorted_df["AvgMinutesPlayed"] = sorted_df["msPlayed"] / 60000
        sorted_df.drop(["msPlayed"], axis=1, inplace=True)
        df_small = sorted_df[["hour", "AvgMinutesPlayed"]]
        val = df_small.values.tolist()
        x = [i for i in range(24)]
        y = [i for i in range(24)]
        for lis in val:
            y[int(lis[0])] = lis[1]
        curDf = pd.DataFrame({"hour": x, "AvgMinutesPlayed": y})
        if plot:
            self.display_hour(curDf)
        return curDf


if __name__ == "__main__":
    df = UserHistory(
        [
            "/Users/iansnyder/Desktop/Projects/Spotify_Proj/Data/ExampleData/StreamingHistory0.json"
        ],
        id="Test User 0",
    )
    df.run()
