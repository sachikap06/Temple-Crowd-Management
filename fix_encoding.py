"""
Fix remaining garbled Unicode in dashboard.py.
Maps each broken sequence (as it appears in the file) to the correct character.
"""

with open('dashboard.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Map: garbled string -> correct replacement
# These are the actual character sequences found in the file
replacements = {
    # Emojis that are partially garbled (some already partially fixed)
    '\u00f0\u0178\u203a\u2022': '\U0001F6D5',  # 🛕
    '\u00f0\u0178\u203a\ufe0f': '\U0001F3DB',  # 🏛
    '\u00f0\u0178\u201c\u0160': '\U0001F4CA',  # 📊
    '\u00f0\u0178\u201c\u02c6': '\U0001F4C8',  # 📈
    '\u00f0\u0178\u201c\u2030': '\U0001F4C9',  # 📉
    '\u00f0\u0178\u201c\u2039': '\U0001F4CB',  # 📋
    '\u00f0\u0178\u201c\u00b1': '\U0001F4F1',  # 📱
    '\u00f0\u0178\u201c\u00a5': '\U0001F4E5',  # 📥
    '\u00f0\u0178\u201c\u0153': '\U0001F4DC',  # 📜
    '\u00f0\u0178\u201c': '\U0001F4CC',        # 📌 (partial match, do last)
    '\u00f0\u0178\u00a7\u00a0': '\U0001F9E0',  # 🧠
    '\u00f0\u0178\u00a7\u00aa': '\U0001F9EA',  # 🧪
    '\u00f0\u0178\u0161\u00a8': '\U0001F6A8',  # 🚨
    '\u00f0\u0178\u0178\u00a2': '\U0001F7E2',  # 🟢
    '\u00f0\u0178\u017d\u00a5': '\U0001F3A5',  # 🎥
    '\u00f0\u0178\u017d\u00af': '\U0001F3AF',  # 🎯
    '\u00f0\u0178\u2018\u00a5': '\U0001F465',  # 👥
    '\u00f0\u0178\u2018\u0081': '\U0001F441',  # 👁
    '\u00f0\u0178\u2018\ufe0f': '\U0001F441',  # 👁 (variant)
    '\u00f0\u0178\u201d\u00ae': '\U0001F52E',  # 🔮
    '\u00f0\u0178\u201d\u00b4': '\U0001F534',  # 🔴
    '\u00f0\u0178\u201d\u201d': '\U0001F514',  # 🔔
    '\u00f0\u0178\u201d\u201e': '\U0001F504',  # 🔄
    '\u00f0\u0178\u2014\u00ba\ufe0f': '\U0001F5FA\uFE0F',  # 🗺️
    '\u00f0\u0178\u2014\u2018\ufe0f': '\U0001F5D1\uFE0F',  # 🗑️
    
    # 3-byte symbols
    '\u00e2\u0153\u2026': '\u2705',    # ✅
    '\u00e2\u0152': '\u274C',          # ❌
    '\u00e2\u0161\u2122\ufe0f': '\u2699\uFE0F',  # ⚙️
    '\u00e2\u0161\u00a0\ufe0f': '\u26A0\uFE0F',  # ⚠️  
    '\u00e2\u00ac\u2021\ufe0f': '\u2B07\uFE0F',  # ⬇️
    '\u00e2\u017e\u2022': '\u2795',    # ➕
    '\u00e2\u017e\u201d': '\u279C',    # ➜
    '\u00e2\u2013\u00b6': '\u25B6',    # ▶
    
    # Box drawing / special
    '\u00e2\u201d\u20ac': '\u2500',    # ─ (box drawing horizontal)
    '\u00e2\u20ac\u00a6': '\u2026',    # … (ellipsis)
    '\u00e2\u20ac\u201c': '\u2013',    # – (en dash)
    '\u00e2\u20ac\u201d': '\u2014',    # — (em dash)
}

count = 0
# Sort by length (longest first) to avoid partial matches
for garbled, correct in sorted(replacements.items(), key=lambda x: -len(x[0])):
    if garbled in text:
        n = text.count(garbled)
        text = text.replace(garbled, correct)
        count += n
        print(f"  Fixed {n}x: U+{ord(correct[0]):04X} ({correct.encode('unicode_escape').decode()})")

print(f"\nTotal fixes: {count}")

# Write back
with open('dashboard.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(text)

print("Done! dashboard.py encoding fixed.")
