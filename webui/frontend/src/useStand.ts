import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { History, Snapshot } from "./types";

const MAX_POINTS = 4 * 3600;

export function useStand() {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const history = useRef<History>({ t: [] });
  const [histVersion, setHistVersion] = useState(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let timer: number | undefined;
    const connect = async () => {
      try {
        const h = await api<History>("/api/history");
        history.current = h;
        setHistVersion((v) => v + 1);
      } catch { /* backend not ready yet */ }
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type !== "step") return;
        setSnap(msg.data as Snapshot);
        const row = msg.row as Record<string, number> | null;
        if (row) {
          const h = history.current;
          const tArr = h.t;
          if (tArr.length && row.t < tArr[tArr.length - 1]) {
            // history was reset on the backend: reload
            for (const k of Object.keys(h)) h[k] = [];
          }
          for (const k of Object.keys(row)) {
            if (!h[k]) h[k] = [];
            h[k].push(row[k]);
            if (h[k].length > MAX_POINTS) h[k].shift();
          }
          setHistVersion((v) => v + 1);
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) timer = window.setTimeout(connect, 1000);
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      ws?.close();
    };
  }, []);
  return { snap, connected, history, histVersion };
}
