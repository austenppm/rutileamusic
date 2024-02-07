import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk
import webbrowser
import innertube
import base64
import requests
from openai import OpenAI
import threading
import webview
from ytmusicapi import YTMusic
from serpapi import GoogleSearch
import json

# Initialize OpenAI client
openai_client = OpenAI()

# Initialize Innertube client
innertube_client = innertube.InnerTube("WEB")

# Initialize YTMusic with your headers or authentication file
yt_music = YTMusic('oauth.json')  # Ensure this file is correctly set up

# Your Freeimage.host API key
api_key = ""

def upload_image_to_freeimage(image_path):
    url = "https://freeimage.host/api/1/upload"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    payload = {
        'key': api_key,
        'action': 'upload',
        'source': base64_image,
        'format': 'json'
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200 and response.json().get("status_code") == 200:
        return response.json()["image"]["url"]
    else:
        raise Exception("Failed to upload image.")

def fetch_google_lens_results(image_url):
    params = {
        'api_key': '', 
        'engine': 'google_lens',
        'url': image_url,
        'hl': 'en',
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def extract_titles_from_lens_results(lens_results):
    titles = []
    for match in lens_results.get("visual_matches", []):
        title = match.get("title")
        if title:
            titles.append(title)
    return titles

def analyze_with_openai():
    global uploaded_image_url
    if uploaded_image_url:
        try:
            description, songs_list = analyze_image_with_openai(uploaded_image_url, current_num_songs, current_token_limit)
            
            # Clear the existing content and update with the new songs list and description
            songs_text.config(state=tk.NORMAL)
            songs_text.delete('1.0', tk.END)
            for song in songs_list:
                songs_text.insert(tk.END, f"{song}\n")
            songs_text.config(state=tk.DISABLED)
            description_label.config(text=description)
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze image: {e}")
    else:
        messagebox.showinfo("Info", "Please load an image first.")

def analyze_image_with_openai(image_url, num_songs=5, token_limit=300):
    prompt = f"What’s in this image? Limit the description to 50 words and give me the top {num_songs} songs that will go with it saying Top {num_songs} songs: "
    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        max_tokens=token_limit,
    )
    responses = response.choices[0]
    content = responses.message.content
    description, songs = content.split(f"\nTop {num_songs} songs:\n", 1)
    songs_list = songs.split("\n")
    return description, songs_list

def analyze_image_with_openai_google(image_url, num_songs=5, token_limit=450, lens_titles=[]):
    lens_titles_str = ", ".join(lens_titles[:5])  # Use the first 3 titles for brevity
    prompt = f"What’s in this image? Limit the description to 50 words and give me the top {num_songs} songs that will go with it saying Top {num_songs} songs: Based on the image at {image_url} and related topics {lens_titles_str}"

    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        max_tokens=token_limit,
    )
    responses = response.choices[0]
    content = responses.message.content
    description, songs = content.split(f"\nTop {num_songs} songs:\n", 1)
    songs_list = songs.split("\n")
    return description, songs_list

def check_with_google_lens():
    global image_path
    if image_path:
        image_url = upload_image_to_freeimage(image_path)
        lens_results = fetch_google_lens_results(image_url)
        titles = extract_titles_from_lens_results(lens_results)
        
        # Clear existing content and update with new recommendations
        songs_text.config(state=tk.NORMAL)
        songs_text.delete('1.0', tk.END)
        
        description, songs_list = analyze_image_with_openai_google(image_url, current_num_songs, current_token_limit_google, titles)
        description_label.config(text=description)  # Update the description label
        for i, song in enumerate(songs_list, 1):
            songs_text.insert(tk.END, f"{song}\n")
        songs_text.config(state=tk.DISABLED)

def get_video_url(song_name):
    # Basic cleanup of the title to improve search accuracy
    cleaned_song_name = song_name.split(' | ')[0]  # Split on ' | ' and take the first part
    cleaned_song_name = cleaned_song_name.split(' - ')[0]  # Further split on ' - ' and take the first part
    cleaned_song_name = cleaned_song_name.split(' >>>> ')[0]  # Remove any trailing ' >>>> ' and similar patterns

    search_results = innertube_client.search(query=cleaned_song_name)
    try:
        video_id = search_results['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0].get('videoRenderer', {}).get('videoId')
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        else:
            raise ValueError(f"No video ID found for the song: {cleaned_song_name}")
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error finding video for {cleaned_song_name}: {e}")
        return None

def get_audio_url(song_name):
    search_results = innertube_client.search(query=song_name)
    try:
        video_id = search_results['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0].get('videoRenderer', {}).get('videoId')
        if video_id:
            return f"https://music.youtube.com/watch?v={video_id}"
        else:
            raise ValueError("No video ID found for the song.")
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error finding video for {song_name}: {e}")
        return None

def load_image():
    global image_path, uploaded_image_url  # Add a global variable to store the uploaded image URL
    image_path = filedialog.askopenfilename()
    if image_path:
        img = Image.open(image_path)
        img = resize_image(img)
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Upload the image to FreeImage hosting
        try:
            uploaded_image_url = upload_image_to_freeimage(image_path)  # Store the uploaded image URL
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to upload image: {e}")

def resize_image(img):
    width, height = img.size
    max_size = 500
    aspect_ratio = width / height
    if width > height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def play_video():
    global current_song_name
    if current_song_name:
        youtube_url = get_video_url(current_song_name)
        if youtube_url:
            webview.create_window('YouTube Video', youtube_url, width=800, height=600)
            webview.start()
        else:
            # Search YouTube with the song name and play the first result if no specific video ID is found
            search_query = current_song_name.replace(" ", "+")  # Format the search query by replacing spaces with '+'
            youtube_search_url = f"https://www.youtube.com/results?search_query={search_query}"
            webview.create_window('YouTube Search', youtube_search_url, width=800, height=600)
            webview.start()
    else:
        messagebox.showinfo("Error", "No song selected.")


def play_audio():
    global current_song_name
    if current_song_name:
        youtubemusic_url = get_audio_url(current_song_name)
        if youtubemusic_url:
            webview.create_window('YouTube Music Video', youtubemusic_url, width=800, height=600)
            webview.start()
        else:
            messagebox.showinfo("Error", "Unable to find a video for the selected song.")
    else:
        messagebox.showinfo("Error", "No song selected.")

# Global variables
current_song_name = ""
last_selected_index = None  # To track the last selected song index for highlighting

def update_songs_list(songs_list, description):
    global last_selected_index
    last_selected_index = None  # Reset the last selected index when a new image is loaded
    songs_text.config(state=tk.NORMAL)
    songs_text.delete('1.0', tk.END)
    for song in songs_list:
        songs_text.insert(tk.END, song + '\n')
    songs_text.config(state=tk.DISABLED)
    description_label.config(text=description)  # Update the description label

def on_song_selected(event):
    global current_song_name, last_selected_index
    try:
        index = songs_text.index("@%d,%d" % (event.x, event.y))
        line, _ = index.split('.')
        line_start = f"{line}.0"
        line_end = f"{line}.end"
        selected_text = songs_text.get(line_start, line_end).strip()

        if "Top 5 Results:" in selected_text:
            return  # Ignore clicks on non-selectable lines

        # Extract title from the selected line (remove numbering and video status)
        current_song_name = selected_text.split('. ', 1)[-1].rsplit(' (', 1)[0]

        songs_text.tag_remove("highlight", "1.0", tk.END)  # Remove any previous highlights
        songs_text.tag_add("highlight", line_start, line_end)  # Highlight selected title
        songs_text.tag_config("highlight", background="lightgray")

        last_selected_index = line_start  # Update last selected index
    except Exception as e:
        current_song_name = ""  # Reset if there's an error
        messagebox.showerror("Error", str(e))


current_num_songs = 5  # Initialize with the default number of songs
current_token_limit = 300  # Initialize with the default token limit
current_token_limit_google = 450  # Initialize with the default token limit for Google Lens

def load_more_songs():
    global current_num_songs, image_path, current_token_limit
    current_num_songs += 5  # Request 5 more songs
    current_token_limit += 100  # Increase the token limit by 50
    if image_path:  # Check if an image is already loaded
        threading.Thread(target=process_image, args=(image_path,), daemon=True).start()

def process_image(image_path):
    try:
        image_url = upload_image_to_freeimage(image_path)
        description, songs_list = analyze_image_with_openai(image_url, current_num_songs, current_token_limit)
        
        # Clear the existing content and update with the new songs list
        songs_text.config(state=tk.NORMAL)
        songs_text.delete('1.0', tk.END)
        for i, song in enumerate(songs_list, 1):
            songs_text.insert(tk.END, f"{song}\n")
        songs_text.config(state=tk.DISABLED)
        
        description_label.config(text=description)  # Update the description under the image
    except Exception as e:
        messagebox.showerror("Error", str(e))
        songs_text.config(state=tk.NORMAL)
        songs_text.delete('1.0', tk.END)
        songs_text.config(state=tk.DISABLED)
        description_label.config(text="")  # Clear the description on error

def google_search_and_lens():
    global image_path
    if image_path:
        image_url = upload_image_to_freeimage(image_path)
        lens_results = fetch_google_lens_results(image_url)
        titles = extract_titles_from_lens_results(lens_results)[:5]  # Take top 5 results

        songs_text.config(state=tk.NORMAL)
        songs_text.delete('1.0', tk.END)
        songs_text.insert(tk.END, "Top 5 Results:\n")

        for i, title in enumerate(titles, start=1):
            video_url = get_video_url(title)
            video_status = "(video found)" if video_url else "(no video found)"
            songs_text.insert(tk.END, f"{i}. {title} {video_status}\n")

        songs_text.config(state=tk.DISABLED)
        description_label.config(text="Top Google Lens results")  # Update the description label


# Modify the setup_gui function to include the "Google Search and Lens" button
def setup_gui():
    global root, left_frame, image_label, songs_frame, songs_text, play_button, play_audio_button, description_label
    root = tk.Tk()
    root.title("Image and Music Integration")

    style = ttk.Style(root)
    style.theme_use("clam")

    left_frame = ttk.Frame(root, padding="10")
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    load_image_btn = ttk.Button(left_frame, text="Load Image", command=load_image)
    load_image_btn.pack(pady=10)
    
    analyze_openai_button = ttk.Button(left_frame, text="Analyze with OpenAI", command=analyze_with_openai)
    analyze_openai_button.pack(pady=5)

    check_lens_button = ttk.Button(left_frame, text="Google Lens and OpenAI", command=check_with_google_lens)
    check_lens_button.pack(pady=5)

    google_search_lens_button = ttk.Button(left_frame, text="Google Search and Lens", command=google_search_and_lens)
    google_search_lens_button.pack(pady=5)

    image_label = ttk.Label(left_frame)
    image_label.pack()

    description_label = ttk.Label(left_frame, wraplength=300, justify=tk.CENTER)
    description_label.pack(pady=10)

    songs_frame = ttk.Frame(root, padding="10")
    songs_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    songs_text = scrolledtext.ScrolledText(songs_frame, wrap=tk.WORD, height=10, relief=tk.FLAT, bg="#f0f0f0", font=("Helvetica", 12))
    songs_text.pack(fill=tk.BOTH, expand=True, pady=10)
    songs_text.bind("<Button-1>", on_song_selected)
    songs_text.config(state=tk.DISABLED)

    play_button = ttk.Button(songs_frame, text="Play Using YouTube", command=play_video)
    play_button.pack(pady=5)

    play_audio_button = ttk.Button(songs_frame, text="Play Using YouTube Music", command=play_audio)
    play_audio_button.pack(pady=5)
    
    load_more_songs_button = ttk.Button(songs_frame, text="Load 5 More Songs", command=load_more_songs)
    load_more_songs_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    setup_gui()