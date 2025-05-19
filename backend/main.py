
import os
import json
import datetime
import torch
import torch.nn as nn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, login
import logging

logging.basicConfig(level=logging.INFO)

# Autenticación Hugging Face
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

# Definición del modelo recurrente DQN
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

# Carga del modelo desde Hugging Face Hub
model = DuelingRecurrentDQN()
pth = hf_hub_download(
    repo_id="Carnotrvficante/Model",
    filename="advanced_blackjack_dqn.pth",
    use_auth_token=hf_token,
)
model.load_state_dict(torch.load(pth, map_location="cpu"))
model.eval()

# Archivo JSON para almacenamiento de créditos
DB_FILE = "users.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    return json.load(open(DB_FILE))

def save_db(db):
    json.dump(db, open(DB_FILE, "w"))

def check_and_use(identifier: str) -> bool:
    db = load_db()
    today = str(datetime.date.today())
    user = db.get(identifier, {"date": today, "daily": 0, "credits": 0})
    # Reset diario si cambió la fecha
    if user["date"] != today:
        user = {"date": today, "daily": 0, "credits": user.get("credits", 0)}
    # 20 usos gratis diarios
    if user["daily"] < 20:
        user["daily"] += 1
    # luego consumen créditos precomprados
    elif user.get("credits", 0) > 0:
        user["credits"] -= 1
    else:
        return False
    db[identifier] = user
    save_db(db)
    return True

# Inicializar FastAPI y CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://blackjack-91kceypby-pablomarpars-projects.vercel.app",
        "https://blackjack-app-pink.vercel.app",
        "https://blackjack-nxjl54q37-pablomarpars-projects.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Esquema de petición para /step (ya no incluye user_id)
class StepRequest(BaseModel):
    ps: int
    dc: int
    ua: bool
    h: list[float]

@app.post("/step")
async def step(req: StepRequest, request: Request):
    # Identificador único: IP del cliente
    client_ip = request.client.host
    if not check_and_use(client_ip):
        raise HTTPException(403, "Sin créditos")

    # Validar tamaño de estado oculto
    if len(req.h) != 128:
        logging.error(f"Hidden state incorrect length: {len(req.h)}; expected 128")
        raise HTTPException(400, "Estado oculto h debe tener longitud 128")

    # Preparar tensores
    x = torch.tensor([[req.ps / 32, req.dc / 11, float(req.ua)]], dtype=torch.float32)
    h = torch.tensor(req.h, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q, h2 = model(x, h)
        p = torch.softmax(q, dim=1)[0].tolist()

    decision = "Hit" if p[1] > p[0] else "Stand"
    conf = max(p)
    h_out = h2.squeeze(0).tolist()

    logging.info(f"Client {client_ip}: decision={decision}, conf={conf}")
    return {"decision": decision, "conf": conf, "h": h_out}

@app.post("/add_credits")
async def add_credits(req: dict, request: Request):
    client_ip = request.client.host
    amount = int(req.get("amount", 0))
    # Añadir créditos al identificador (IP)
    full_db = load_db()
    user = full_db.get(client_ip, {"date": str(datetime.date.today()), "daily": 0, "credits": 0})
    user["credits"] = user.get("credits", 0) + amount
    full_db[client_ip] = user
    save_db(full_db)
    return {"status": "ok", "new_credits": user["credits"]}

