import os
import json
import datetime
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, login
import logging

logging.basicConfig(level=logging.INFO)

# Autenticación Hugging Face
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

class DuelingRecurrentDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(3, 128)
        self.value_fc = nn.Linear(128, 1)
        self.adv_fc   = nn.Linear(128, 2)

    def forward(self, x, h):
        h2 = self.gru(x, h)
        v = self.value_fc(h2)
        a = self.adv_fc(h2)
        return v + a - a.mean(1, keepdim=True), h2

model = DuelingRecurrentDQN()
pth = hf_hub_download(
    repo_id="Carnotrvficante/Model",
    filename="advanced_blackjack_dqn.pth",
    use_auth_token=hf_token,
)
model.load_state_dict(torch.load(pth, map_location="cpu"))
model.eval()

DB_FILE = "users.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    return json.load(open(DB_FILE))

def save_db(db):
    json.dump(db, open(DB_FILE, "w"))

def check_and_use(user_id):
    db = load_db()
    today = str(datetime.date.today())
    user = db.get(user_id, {"date": today, "daily": 0, "credits": 0})
    if user["date"] != today:
        user = {"date": today, "daily": 0, "credits": user.get("credits", 0)}
    if user["daily"] < 20:
        user["daily"] += 1
    elif user["credits"] > 0:
        user["credits"] -= 1
    else:
        return False
    db[user_id] = user
    save_db(db)
    return True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://blackjack-91kceypby-pablomarpars-projects.vercel.app",
        "https://blackjack-app-pink.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StepRequest(BaseModel):
    ps: int
    dc: int
    ua: bool
    h: list[float]
    user_id: str

@app.post("/step")
async def step(req: StepRequest):
    if not check_and_use(req.user_id):
        raise HTTPException(403, "Sin créditos")

    if len(req.h) != 128:
        logging.error(f"Hidden state incorrect length: {len(req.h)}; expected 128")
        raise HTTPException(400, "Estado oculto h debe tener longitud 128")

    x = torch.tensor([[req.ps / 32, req.dc / 11, float(req.ua)]], dtype=torch.float32)  # shape (1,3)
    h = torch.tensor(req.h, dtype=torch.float32).unsqueeze(0)  # shape (1,128)

    with torch.no_grad():
        q, h2 = model(x, h)  # h2 shape (1,128)
        p = torch.softmax(q, dim=1)[0].tolist()

    decision = "Hit" if p[1] > p[0] else "Stand"
    conf = max(p)
    h_out = h2.squeeze(0).tolist()  # devolver vector plano

    logging.info(f"User {req.user_id}: decision={decision}, conf={conf}")

    return {
        "decision": decision,
        "conf": conf,
        "h": h_out
    }

@app.post("/add_credits")
async def add_credits(req: dict):
    user = req.get("user_id")
    amount = int(req.get("amount", 0))
    db = load_db()
    u = db.get(user, {"date": str(datetime.date.today()), "daily": 0, "credits": 0})
    u["credits"] = u.get("credits", 0) + amount
    db[user] = u
    save_db(db)
    return {"status": "ok", "new_credits": u["credits"]}
