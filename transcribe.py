"""Free automatic call transcriber (runs on GitHub Actions, no browser, no paid API).

1. Reads the live queue that Odoo cron 261 publishes (attachment 225588):
   pending RingingGo calls with talk time >= 20 s, newest first.
2. Downloads each recording and transcribes it with open-source Whisper (faster-whisper, CPU).
3. Posts {id, text} to the Odoo "Call transcripts" webhook (server action 1531).
   One line per speech segment, so the AI pass can label Bot / Agent / Customer.

Logs print counts only (no names, numbers or transcript text).
"""
import json, os, re, sys, tempfile, time, urllib.request

QUEUE_URL = os.environ["QUEUE_URL"]
HOOK_URL = os.environ["HOOK_URL"]
MODEL = os.environ.get("WHISPER_MODEL", "small")
LANG = os.environ.get("WHISPER_LANG", "en") or None
BUDGET = int(os.environ.get("TIME_BUDGET_SEC", "2700"))
MAX_CALLS = int(os.environ.get("MAX_CALLS", "400"))
UA = {"User-Agent": "Mozilla/5.0 (call-transcriber)"}
BEAM = int(os.environ.get("WHISPER_BEAM", "5"))
# Business words Whisper otherwise mishears (branches, brands, services, payment apps).
VOCAB = os.environ.get("WHISPER_VOCAB") or (
    "Pet Corner, Vet Veterinary Clinic, Value Pets, Handyman. Branches: Al Barsha, Nshama, DIP, "
    "Motor City, Jumeirah Park, JVC, JBR, Al Warsan, Meydan Heights, Nad Al Hamar, Mirdif, "
    "Sheikh Zayed Road, Business Bay, Fujairah, Khalifa City, Abu Dhabi, Sharjah, Al Mamsha, Al Ain. "
    "Grooming, mobile grooming, full grooming, vaccination, rabies, deworming, neutering, boarding. "
    "Royal Canin, Hill's, Purina, Smudges, Van Cat, Amanova, Orijen, Acana, Applaws, Whiskas, Kit Cat, "
    "Schesir, Monge, Farmina, autoship, WhatsApp, Tabby, Tamara, Groupon, Talabat, Noon, Instashop.")
ENGINE = "whisper-" + os.environ.get("WHISPER_MODEL", "small")
start = time.time()


def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def post(items):
    body = json.dumps({"items": items}).encode()
    req = urllib.request.Request(HOOK_URL, data=body, headers={**UA, "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60).status


def load_queue():
    js = get(QUEUE_URL + "&t=%d" % time.time()).decode()
    m = re.search(r"__CQ_LIVE=(\[.*?\]);", js, re.S)
    return json.loads(m.group(1)) if m else []


def main():
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 2)
    done, empty, fail, audio_sec = 0, 0, 0, 0.0
    seen = set()
    queue = load_queue()
    print("queue:", len(queue), "model:", MODEL)
    last_refresh = time.time()
    while time.time() - start < BUDGET and done + empty < MAX_CALLS:
        if time.time() - last_refresh > 600:
            queue, last_refresh = load_queue(), time.time()
        item = next((q for q in queue if q[0] not in seen), None)
        if not item:
            break
        cid, url = item[0], item[1]
        seen.add(cid)
        try:
            data = get(url, timeout=120)
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                f.write(data); f.flush()
                segs, info = model.transcribe(
                    f.name, language=LANG, beam_size=BEAM, best_of=BEAM,
                    vad_filter=True, vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 300},
                    initial_prompt="Customer service call, Pet Corner Dubai.",
                    hotwords=VOCAB, condition_on_previous_text=False,
                    no_speech_threshold=0.5, compression_ratio_threshold=2.2)
                lines = [s.text.strip() for s in segs if s.text.strip()]
                audio_sec += info.duration
            text = "\n".join(lines)
            post([{"id": cid, "text": text or "(no clear speech recognised)", "engine": ENGINE}])
            if text:
                done += 1
            else:
                empty += 1
        except Exception as e:  # keep going; failures are retried next run
            fail += 1
            print("fail:", cid, type(e).__name__)
    took = time.time() - start
    print("transcribed:", done, "empty:", empty, "failed:", fail,
          "audio_min: %.1f" % (audio_sec / 60), "run_min: %.1f" % (took / 60),
          "speed: %.1fx" % (audio_sec / took if took else 0))


if __name__ == "__main__":
    sys.exit(main())
