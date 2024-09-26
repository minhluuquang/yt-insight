# Discord Bot with discord.py, openai, and Poetry

## Project Overview
This Discord bot is designed to summarize YouTube videos and extract topics and insights using chain-of-thought reasoning. The bot listens for the `/sum` command in Discord, fetches the transcript of the provided YouTube video, and processes it to generate a summary and related insights.

## Prerequisites
### Required Software and Libraries
- **Python**: Ensure you have Python 3.8 or higher installed.
- **Poetry**: For dependency management and packaging.
- **discord.py**: Python library for interacting with the Discord API.
- **openai**: Library for processing and summarizing text.

### External APIs
- **OpenAI API**: Required for generating summaries and insights.
  - Follow the [OpenAI API setup guide](https://platform.openai.com/docs/api-reference/introduction) to obtain your API key.
- **Discord API**: Required for interacting with the Discord API.
  - Follow the [Discord API setup guide](https://discord.com/developers/docs/intro) to obtain your API key.

## Installation
### Step-by-Step Guide
1. **Clone the Repository**
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install Dependencies using Poetry**
    ```sh
    poetry install
    ```

3. **Set Environment Variables**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    DISCORD_SECRET_TOKEN=your_discord_secret_token
    GUILD_ID=your_guild_id
    OPENAI_API_KEY=your_openai_api_key
    ```

## Running the Bot
### Starting the Bot
To start the bot, run:
```sh
poetry run dev
```

## Usage
### Commands
- `/sum <YouTube URL>`: Summarize the YouTube video and extract topics and insights.
