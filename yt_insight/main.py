import os
import dspy
from dotenv import load_dotenv
from yt_insight.bot import discord_client


load_dotenv()


def main():
    # Run the bot
    discord_client.run(os.getenv("DISCORD_SECRET_TOKEN"))


if __name__ == "__main__":
    main()
