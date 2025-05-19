const backend = "https://assistant-blackjack-ai.onrender.com";
let h = new Array(128).fill(0);

// Pestañas con resaltado de activa
document.querySelectorAll("nav button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach(s => s.classList.add("hidden"));
    document.getElementById(btn.dataset.tab).classList.remove("hidden");
    document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  };
});

// Predecir jugada
document.getElementById("stepBtn").onclick = async () => {
  const ps = +document.getElementById("ps").value;
  const dc = +document.getElementById("dc").value;
  const ua = document.getElementById("ua").checked;

  const stepBtn = document.getElementById("stepBtn");
  stepBtn.innerText = "Pensando…";
  stepBtn.disabled = true;

  try {
    const res = await fetch(backend + "/step", {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ps, dc, ua, h })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const j = await res.json();
    document.getElementById("decisionOut").innerText = j.decision;
    document.getElementById("confidenceOut").innerText = (j.conf * 100).toFixed(2) + "%";
    h = j.h;
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    stepBtn.innerText = "Predecir jugada";
    stepBtn.disabled = false;
  }
};

// Nueva mano
document.getElementById("resetBtn").onclick = () => {
  h = new Array(128).fill(0);
  document.getElementById("decisionOut").innerText = "—";
  document.getElementById("confidenceOut").innerText = "—";
};

// Comprar créditos
const buy = amount => async () => {
  const res = await fetch(backend + "/add_credits", {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount })
  });
  const j = await res.json();
  document.getElementById("creditStatus").innerText = JSON.stringify(j, null, 2);
};
document.getElementById("buy50").onclick = buy(50);
document.getElementById("buy1000").onclick = buy(1000);
