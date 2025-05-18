import os
import json
import datetime
import torch, torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, login

# ─── Autoauth HF ─────────────────────────────────
hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)

# ─── Modelo ──────────────────────────────────────
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
pth = hf_hub_download("Carnotrvficante/Model", "advanced_blackjack_dqn.pth", use_auth_token=hf_token)
model.load_state_dict(torch.load(pth, map_location="cpu"))
model.eval()

# ─── Créditos (misma lógica que credit_manager) ──
DB_FILE = "users.json"
def load_db():
    if not os.path.exists(DB_FILE): return {}
    return json.load(open(DB_FILE))
def save_db(d): json.dump(d, open(DB_FILE,"w"))
def check_and_use(user_id):
    db = load_db()
    today = str(datetime.date.today())
    u = db.get(user_id, {"date":today,"daily":0,"credits":0})
    if u["date"] != today:
        u = {"date":today,"daily":0,"credits":u["credits"]}
    if u["daily"] < 20:
        u["daily"]+=1
    elif u["credits"]>0:
        u["credits"]-=1
    else:
        return False
    db[user_id]=u; save_db(db)
    return True

# ─── FastAPI ─────────────────────────────────────
app = FastAPI()

class Request(BaseModel):
    ps: int
    dc: int
    ua: bool
    h: list[float]
    user_id: str

@app.post("/step")
def step(req: Request):
    if not check_and_use(req.user_id):
        raise HTTPException(403, "Sin créditos")
    x = torch.tensor([[req.ps/32, req.dc/11, float(req.ua)]])
    h = torch.tensor(req.h)
    with torch.no_grad():
        q, h2 = model(x, h)
        p = torch.softmax(q,1)[0].tolist()
    return {"decision": "Hit" if p[1]>p[0] else "Stand",
            "conf": max(p),
            "h": h2.tolist()}

@app.post("/add_credits")
def add_credits(req: dict):
    user = req["user_id"]; amt = int(req["amount"])
    db = load_db(); u = db.get(user,{"date":str(datetime.date.today()),"daily":0,"credits":0})
    u["credits"] = u.get("credits",0) + amt; db[user]=u; save_db(db)
    return {"status":"ok"}
