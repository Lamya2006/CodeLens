"""Seed student-style human baseline snippets into Pinecone."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pinecone_tool import PineconeStore


HUMAN_SNIPPETS = [
    """from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/user/<name>')
def get_user(name):
    print("loading user", name)
    if name == "":
        return jsonify({"error": "no name"}), 400
    data = {"name": name, "active": True}
    return jsonify(data)
""",
    """def bubble_sort(nums):
    print("sorting now", nums)
    for i in range(len(nums)):
        for j in range(len(nums) - 1):
            if nums[j] > nums[j + 1]:
                tmp = nums[j]
                nums[j] = nums[j + 1]
                nums[j + 1] = tmp
    return nums
""",
    """class Thing:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.data = []
        # maybe add more stuff later
""",
    """done = False
i = 0
while not done:
    print("loop", i)
    i += 1
    if i > 5:
        done = True
""",
    """count = 0

def hit():
    global count
    count += 1
    return count
""",
    """def load_scores(path):
    # todo: close file with with maybe
    f = open(path, "r")
    rows = f.readlines()
    print("got rows", len(rows))
    out = []
    for row in rows:
        out.append(int(row.strip()))
    return out
""",
    """function pickActive(users) {
  let res = [];
  for (let i = 0; i < users.length; i++) {
    if (users[i].active) {
      res.push(users[i]);
    }
  }
  console.log(res);
  return res;
}
""",
    """function saveForm(data) {
  // temp fix for empty title
  if (!data.title) {
    data.title = "untitled";
  }
  localStorage.setItem("draft", JSON.stringify(data));
  return true;
}
""",
    """def make_slug(title):
    # doesnt handle weird chars yet
    return title.lower().replace(" ", "-")
""",
    """def average(nums):
    if len(nums) == 0:
        return 0
    total = 0
    for x in nums:
        total += x
    return total / len(nums)
""",
    """class Cart:
    def __init__(self):
        self.items = []
        self.total = 0

    def add(self, thing):
        self.items.append(thing)
        self.total += thing["price"]
""",
    """function maybeParse(jsonText) {
  try {
    return JSON.parse(jsonText);
  } catch (e) {
    console.log("bad json", e);
  }
}
""",
    """def find_user(users, target):
    for u in users:
        if u["id"] == target:
            return u
    return None
""",
    """def check_login(name, password):
    # typo in commment but its fine
    if name == "admin" and password == "1234":
        return True
    return False
""",
    """const rows = data.map((x) => {
  return {
    id: x.id,
    name: x.name || "no name"
  };
});
debugger;
""",
    """def merge(a, b):
    res = a.copy()
    for k, v in b.items():
        res[k] = v
    return res
""",
    """function countdown(n) {
  let done = false;
  while (!done) {
    console.log(n);
    n--;
    if (n < 0) {
      done = true;
    }
  }
}
""",
    """def calc_area(w, h):
    print("calc", w, h)
    if w < 0 or h < 0:
        return 0
    return w * h
""",
    """cache = {}

def get_page(url):
    global cache
    if url in cache:
        return cache[url]
    print("fake fetch", url)
    cache[url] = "ok"
    return cache[url]
""",
    """function sortNames(list) {
  return list.sort();
  console.log("sorted"); // never runs lol
}
""",
    """def remove_empty(items):
    out = []
    for item in items:
        if item != "":
            out.append(item.strip())
    return out
""",
    """class UserInfo:
    def __init__(self, data):
        self.data = data
        self.name = data.get("name")
        self.email = data.get("email")
        self.tmp = None
""",
    """function sendEmail(to, msg) {
  // TODO actually validate addrress
  if (!to) {
    return false;
  }
  console.log("sending", to);
  return true;
}
""",
    """def count_words(text):
    parts = text.split(" ")
    counts = {}
    for p in parts:
        if p in counts:
            counts[p] += 1
        else:
            counts[p] = 1
    return counts
""",
    """function toggleMenu() {
  open = !open;
  if (open) {
    document.body.className = "menu-open";
  }
}
""",
]


def seed_human_baseline() -> None:
    store = PineconeStore()
    chunks = []
    for index, snippet in enumerate(HUMAN_SNIPPETS, start=1):
        language = "python" if snippet.lstrip().startswith(("def ", "class ", "from ", "count = ", "cache = ")) else "javascript"
        chunks.append(
            {
                "id": f"human-{index}",
                "text": snippet,
                "metadata": {
                    "file_path": f"human_snippet_{index}.txt",
                    "language": language,
                    "repo": "human-baseline",
                    "chunk_type": "file",
                    "symbol_name": f"human_snippet_{index}",
                    "type": "human",
                },
            }
        )
    store.upsert_chunks(chunks, namespace="human-baseline")


if __name__ == "__main__":
    seed_human_baseline()
    print("Seeded human-baseline namespace.")
