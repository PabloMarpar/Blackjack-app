const backend = "https://TU_BACKEND_URL";
let userId = null;

// Login
document.getElementById("loginBtn").onclick = () => {
  const input = document.getElementById("userId");
  userId = input.value.trim().toLowerCase();
  if (!userId) return alert("Introduce tu ID");
  document.getElementById("login").classList.add("hidden");
  document.getElementById("app").classList.remove("hidden");
};

// Tabs
document.querySelectorAll("nav button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach(s => s.classList.add("hidden"));
    document.getElementById(btn.dataset.tab).classList.remove("hidden");
  };
});

// Jugar paso
document.getElementById("stepBtn").onclick = async () => {
  const ps = +document.getElementById("ps").value;
  const dc = +document.getElementById("dc").value;
  const ua = document.getElementById("ua").checked;
  const h = Array(128).fill(0);
  try {
    const res = await fetch(backend + "/step", {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ps, dc, ua, h, user_id: userId })
    });
    const j = await res.json();
    document.getElementById("out").innerText = JSON.stringify(j, null, 2);
  } catch (e) {
    alert("Error: " + e.message);
  }
};

// Comprar créditos
const buy = (amount) => async () => {
  const res = await fetch(backend + "/add_credits", {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({ user_id: userId, amount })
  });
  const j = await res.json();
  document.getElementById("creditStatus").innerText = JSON.stringify(j, null, 2);
};

document.getElementById("buy50").onclick = buy(50);
document.getElementById("buy1000").onclick = buy(1000);