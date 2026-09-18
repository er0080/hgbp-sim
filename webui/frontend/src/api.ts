export async function api<T = any>(path: string, body?: any, method?: string): Promise<T> {
  const r = await fetch(path, {
    method: method ?? (body !== undefined ? "POST" : "GET"),
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
export const fmt = (v: number | undefined, d = 1) => (v === undefined || Number.isNaN(v) ? "--" : v.toFixed(d));
export const hms = (t: number) => {
  const s = Math.max(0, Math.floor(t));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
};
