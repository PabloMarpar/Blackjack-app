const backend = "https://assistant-blackjack-ai.onrender.com";
let userId = null;
let h = new Array(128).fill(0);  // 1) Estado oculto persistente

// Login
document.getElementById("loginBtn").onclick = () => {
  userId = document.getElementById("userId").value.trim().toLowerCase();
  if (!userId) return alert("Introduce tu ID");
  document.getElementById("login").classList.add("hidden");
  document.getElementById("app").classList.remove("hidden");
};

// Pestañas con resaltado de activa
document.querySelectorAll("nav button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach(s => s.classList.add("hidden"));
    document.getElementById(btn.dataset.tab).classList.remove("hidden");
    document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  };
});

// Jugar paso
document.getElementById("stepBtn").onclick = async () => {
  const ps = +document.getElementById("ps").value;
  const dc = +document.getElementById("dc").value;
  const ua = document.getElementById("ua").checked;

  // 3) Indicador de carga
  const stepBtn = document.getElementById("stepBtn");
  stepBtn.innerText = "Pensando…";
  stepBtn.disabled = true;

  try {
    const res = await fetch(backend + "/step", {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ps, dc, ua, h, user_id: userId })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const j = await res.json();
    document.getElementById("out").innerText = JSON.stringify(j, null, 2);

    h = j.h;  // 1) Actualizar estado oculto tras cada paso
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    stepBtn.innerText = "Jugar paso";
    stepBtn.disabled = false;
  }
};

// 2) Reset mano
document.getElementById("resetBtn").onclick = () => {
  h = new Array(128).fill(0);
  document.getElementById("out").innerText = "";
};

// Comprar créditos
const buy = amount => async () => {
  const res = await fetch(backend + "/add_credits", {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({ user_id: userId, amount })
  });
  const j = await res.json();
  document.getElementById("creditStatus").innerText = JSON.stringify(j, null, 2);
};
document.getElementById("buy50").onclick  = buy(50);
document.getElementById("buy1000").onclick = buy(1000);
