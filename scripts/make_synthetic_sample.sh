#!/usr/bin/env bash
# Build samples/synthetic_th.flac: a ~75 s scripted Thai meeting with 3 distinct "voices".
# macOS only (uses `say -v Kanya`). Voices are faked by pitch-shifting Kanya with ffmpeg
# (asetrate changes pitch+speed, atempo restores speed):
#   A = Kanya as-is, B = pitch x0.80 (lower), C = pitch x1.25 (higher).
# Ground truth: A = สมชาย (PM), B = มาลี (marketing), C = อนันต์ (dev lead).
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="samples/synthetic_th.flac"
TRUTH="samples/synthetic_th.truth.tsv"  # speaker<TAB>start_s<TAB>end_s<TAB>text
RATE=24000
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

LINES=(
  "A|สวัสดีทุกคน ฉันชื่อสมชาย เป็นผู้จัดการโครงการ วันนี้เรามาคุยกันเรื่องกำหนดเปิดตัวแอปใหม่"
  "B|สวัสดี ฉันชื่อมาลี ดูแลฝ่ายการตลาด"
  "C|สวัสดี ฉันชื่ออนันต์ เป็นหัวหน้าทีมพัฒนา"
  "A|อนันต์ ตอนนี้ความคืบหน้าของแอปเป็นอย่างไรบ้าง"
  "C|ฟีเจอร์หลักเสร็จแล้วแปดสิบเปอร์เซ็นต์ เหลือระบบแจ้งเตือนกับการทดสอบความปลอดภัย คาดว่าจะเสร็จภายในสองสัปดาห์"
  "B|ฝ่ายการตลาดอยากเปิดตัววันที่สิบห้าของเดือนหน้า จะทันไหม"
  "C|ถ้าไม่มีปัญหาแทรก น่าจะทัน แต่ต้องรอผลทดสอบความปลอดภัยก่อน"
  "A|งั้นเราตัดสินใจเลื่อนวันเปิดตัวเป็นวันที่ยี่สิบของเดือนหน้า เพื่อให้มีเวลาทดสอบมากขึ้น ทุกคนเห็นด้วยไหม"
  "B|เห็นด้วย ฉันจะปรับแผนโฆษณาให้ตรงกับวันใหม่"
  "C|เห็นด้วยเหมือนกัน"
  "A|ดีมาก มาลี ช่วยส่งแผนโฆษณาฉบับใหม่ให้ทุกคนภายในวันศุกร์นี้ด้วยนะ"
  "B|ได้เลย ส่งภายในวันศุกร์"
  "A|อนันต์ ช่วยรายงานผลทดสอบความปลอดภัยในการประชุมวันพุธหน้า"
  "C|รับทราบ"
  "A|มีใครมีคำถามอะไรอีกไหม"
  "B|มีหนึ่งข้อ งบโฆษณาออนไลน์ได้รับอนุมัติหรือยัง"
  "A|ยังไม่ได้รับอนุมัติ ฉันจะไปถามฝ่ายการเงิน แล้วแจ้งให้ทราบ"
  "A|ขอบคุณทุกคน ประชุมวันนี้จบเท่านี้"
)

pitch_factor() { case "$1" in A) echo 1.0 ;; B) echo 0.80 ;; C) echo 1.25 ;; esac; }

# 0.5 s of silence between turns
ffmpeg -loglevel error -y -f lavfi -i "anullsrc=r=$RATE:cl=mono" -t 0.5 "$TMP/gap.wav"

: > "$TMP/list.txt"
: > "$TRUTH"
t=0
i=0
for entry in "${LINES[@]}"; do
  spk="${entry%%|*}"; text="${entry#*|}"
  f="$(pitch_factor "$spk")"
  n="$(printf '%02d' "$i")"
  say -v Kanya -o "$TMP/raw$n.aiff" "$text"
  inv="$(awk -v f="$f" 'BEGIN { printf "%.4f", 1 / f }')"
  ffmpeg -loglevel error -y -i "$TMP/raw$n.aiff" -ac 1 -ar "$RATE" \
    -af "asetrate=$RATE*$f,aresample=$RATE,atempo=$inv" "$TMP/line$n.wav"
  dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$TMP/line$n.wav")"
  end="$(awk -v t="$t" -v d="$dur" 'BEGIN { printf "%.2f", t + d }')"
  printf '%s\t%.2f\t%s\t%s\n' "$spk" "$t" "$end" "$text" >> "$TRUTH"
  t="$(awk -v e="$end" 'BEGIN { printf "%.2f", e + 0.5 }')"
  echo "file 'line$n.wav'" >> "$TMP/list.txt"
  echo "file 'gap.wav'" >> "$TMP/list.txt"
  i=$((i + 1))
done

ffmpeg -loglevel error -y -f concat -safe 0 -i "$TMP/list.txt" -ac 1 -ar 16000 "$OUT"
DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")"
echo "wrote $OUT (${DUR}s)"
