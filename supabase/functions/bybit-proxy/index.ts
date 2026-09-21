// Signed Bybit read-proxy (2026-09-21) - see project_bybit_geoblock_proxy
// memory / README. GitHub Actions runners got a confirmed 403 from Bybit's
// CloudFront: "configured to block access from your country" - not an IP
// whitelist or credentials problem, a geo-block on wherever GitHub's
// hosted runners happen to egress from. Supabase's own infra (AWS
// eu-west-1) was confirmed NOT blocked (200 OK on a public Bybit
// endpoint), so this function signs and forwards the SAME read-only
// wallet-balance requests bybit_balance.py already made directly - the
// parsing/business logic stays in Python (bybit_balance.py), this is a
// pure signing+forwarding proxy so there's nothing here to drift out of
// sync with the real logic.
//
// Locked down to exactly the two read-only endpoints bybit_balance.py
// uses - GET only, no order-placement path exists here, matching that
// module's own "read-only, structurally incapable of trading" design.
//
// Requires these Edge Function secrets:
//   BYBIT_PROXY_SECRET - shared secret, checked against X-Proxy-Secret.
//   BYBIT_API_KEY / BYBIT_API_SECRET - the same real-mainnet READ-ONLY key
//     bybit_balance.py uses locally/in GitHub Secrets.

const PROXY_SECRET = Deno.env.get("BYBIT_PROXY_SECRET")!;
const API_KEY = Deno.env.get("BYBIT_API_KEY")!;
const API_SECRET = Deno.env.get("BYBIT_API_SECRET")!;
const BASE_URL = "https://api.bybit.com";
const RECV_WINDOW = "20000";

const ALLOWED_PATHS = [
  "/v5/account/wallet-balance",
  "/v5/asset/transfer/query-account-coins-balance",
];

async function sign(payload: string, timestamp: string): Promise<string> {
  const raw = `${timestamp}${API_KEY}${RECV_WINDOW}${payload}`;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(API_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(raw));
  return Array.from(new Uint8Array(sig)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

Deno.serve(async (req) => {
  if (req.headers.get("X-Proxy-Secret") !== PROXY_SECRET) {
    return new Response("forbidden", { status: 403 });
  }

  let payload: { path?: string; params?: Record<string, string> };
  try {
    payload = await req.json();
  } catch {
    return new Response("bad request", { status: 400 });
  }

  const path = payload.path ?? "";
  if (!ALLOWED_PATHS.includes(path)) {
    return new Response("path not allowed", { status: 400 });
  }

  const params = payload.params ?? {};
  const query = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join("&");

  const timestamp = Date.now().toString();
  const signature = await sign(query, timestamp);
  const url = `${BASE_URL}${path}${query ? "?" + query : ""}`;

  const resp = await fetch(url, {
    headers: {
      "X-BAPI-API-KEY": API_KEY,
      "X-BAPI-TIMESTAMP": timestamp,
      "X-BAPI-RECV-WINDOW": RECV_WINDOW,
      "X-BAPI-SIGN": signature,
    },
  });

  const body = await resp.text();
  return new Response(body, {
    status: resp.status,
    headers: { "Content-Type": "application/json" },
  });
});
