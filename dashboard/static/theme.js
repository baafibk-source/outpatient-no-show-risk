// Applies the saved colour theme before first paint, so dark mode never flashes light.
// "auto" (or nothing saved) leaves the choice to the system setting.
try {
  const t = localStorage.getItem("dna-theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch {}
