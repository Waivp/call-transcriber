# call-transcriber

Free, open-source speech-to-text runner (faster-whisper) on GitHub Actions.
It reads a private queue of call-recording links, transcribes them, and posts the text back to a private webhook.

- No customer data, recordings or transcripts are stored in this repository.
- The queue and webhook addresses are GitHub Actions **secrets** (`QUEUE_URL`, `HOOK_URL`); logs print counts only.
