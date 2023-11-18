import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from libs.data_processing import (
    generate_timestamps_from_yt_transcript,
    convert_timestamps_to_intervals,
    convert_transcript_to_dict,
    extract_transcript_from_timeframe,
)

INPUT_FILE_PATH = "data/episode_urls.txt"
OUTPUT_FILE_PATH = "data/andrew_huberman_episodes.json"


if __name__ == "__main__":
    with open(INPUT_FILE_PATH, "r") as file:
        urls = [line.strip() for line in file]

    # urls = [
    #     "https://www.hubermanlab.com/episode/ama-1-leveraging-ultradian-cycles-how-to-protect-your-brain-seed-oils-examined-and-more",
    #     "https://www.hubermanlab.com/episode/ama-10-benefits-of-nature-grounding-hearing-loss-research-avoiding-altitude-sickness",
    # ]

    items = []

    for url in tqdm(urls, total=len(urls)):
        transcripts = []
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                hero_parent = soup.find("header", id="hero-section")

                # Get date
                date = hero_parent.find("div", class_="episode-chip-date-wrapper")
                date = date.find("div", class_="u-line-height-none").text

                # Get title
                title = hero_parent.find("h1", class_="h3").text

                # Get topics
                topics = hero_parent.find_all("a", class_="chip-topics")
                topics = [topic.text for topic in topics]

                # Get timestamps
                ts_parent = soup.find("div", id="timestamps-section")
                timestamps = ts_parent.find_all("li")
                timestamps = [ts.text for ts in timestamps]
                intervals = convert_timestamps_to_intervals(timestamps)

                # Get youtube
                youtube = hero_parent.find("div", class_="platforms-chips-wrapper")
                youtube = youtube.find_all("a")
                youtube = [youtube["href"] for youtube in youtube]
                yt_url = [youtube for youtube in youtube if "youtu" in youtube][0]
                yt_transcripts = convert_transcript_to_dict(yt_url)

                # If intervals exist, extract transcript for each interval
                if len(intervals) > 0:
                    for interval in intervals:
                        transcript = extract_transcript_from_timeframe(
                            yt_transcripts,
                            float(interval["start"]),
                            float(interval["end"]),
                        )
                        interval["transcript"] = transcript
                        transcripts.append(interval)
                # Otherwise, extract transcript for entire video
                else:
                    transcripts = generate_timestamps_from_yt_transcript(yt_transcripts)

                # Get description
                desc_parent = soup.find("section", id="details")
                description = desc_parent.find("p").text

                # Get guest
                guest_parent = soup.find("div", id="tab-content")
                guest = guest_parent.find("div", class_="guest-card")
                if guest:
                    guest = guest.find("h3").text.strip()
                else:
                    guest = None

                items.append(
                    {
                        "url": url,
                        "created": date,
                        "title": title,
                        "topics": topics,
                        "guest": guest,
                        "youtube": yt_url,
                        "transcripts": transcripts,
                        "description": description,
                    }
                )
            else:
                items.append(
                    {
                        "url": url,
                        "created": None,
                        "title": None,
                        "topics": None,
                        "guest": None,
                        "youtube": None,
                        "transcripts": None,
                        "description": None,
                    }
                )
                print(
                    f"Failed to retrieve content from {url}, status code: {response.status_code}"
                )
        except Exception as e:
            items.append(
                {
                    "url": url,
                    "created": None,
                    "title": None,
                    "topics": None,
                    "guest": None,
                    "youtube": None,
                    "transcripts": None,
                    "description": None,
                }
            )
            print(f"An error occurred while processing {url}: {e}")

    # print(items)

    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(items, file, ensure_ascii=False, indent=4)
