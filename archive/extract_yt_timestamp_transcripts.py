from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import re
import urllib.parse as urlparse


def convert_transcript_to_dict(url):
    """
    Convert YouTube video transcript to a dictionary.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        dict: A dictionary where the keys are the timestamps of each transcript
        entry and the values are the corresponding transcript text.
    """
    parsed_url = urlparse.urlparse(url)
    query_params = urlparse.parse_qs(parsed_url.query)
    video_id = query_params["v"][0]
    data = YouTubeTranscriptApi.get_transcript(video_id)

    result = {}
    for entry in data:
        if "text" in entry:
            result[entry["start"]] = entry["text"]
    return result


def extract_transcript_from_timeframe(data_dict, start, end):
    """
    Extracts the transcript of a given timeframe from a dictionary of YouTube video transcripts.

    Args:
    data_dict (dict): A dictionary containing the YouTube video transcript data.
    start (float): The start time of the desired transcript in seconds.
    end (float): The end time of the desired transcript in seconds.

    Returns:
    str: The transcript of the given timeframe.
    """

    def clean_test(text):
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" -", "-")
        text = text.strip()
        return text

    keys = np.array(list(data_dict.keys()))
    closest_start = np.argmin(np.abs(keys - start))
    closest_end = np.argmin(np.abs(keys - end))
    timestamps = keys[closest_start : closest_end + 1]

    transcript = ""
    for t in timestamps:
        text = clean_test(data_dict[t])
        transcript += text + " "

    return transcript


url = "https://www.youtube.com/watch?v=6ZrlsVx85ek"

res = convert_transcript_to_dict(url)
transcript = extract_transcript_from_timeframe(res, 400, 1000)

print(transcript)
