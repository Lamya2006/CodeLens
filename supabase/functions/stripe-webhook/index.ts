import { createClient } from "jsr:@supabase/supabase-js@2";
import Stripe from "npm:stripe@17";

const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY") ?? "");
const webhookSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET") ?? "";
const supabase = createClient(
  Deno.env.get("SUPABASE_URL") ?? "",
  Deno.env.get("DB_SERVICE_ROLE_KEY") ?? "",
);

Deno.serve(async (req) => {
  const signature = req.headers.get("stripe-signature");
  if (!signature) {
    return new Response("Missing stripe-signature header", { status: 400 });
  }

  const body = await req.text();

  let event: Stripe.Event;
  try {
    event = await stripe.webhooks.constructEventAsync(body, signature, webhookSecret);
  } catch (err) {
    console.error("Webhook signature verification failed:", err);
    return new Response("Invalid signature", { status: 400 });
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object as Stripe.Checkout.Session;
    const username = session.metadata?.github_username;

    if (!username) {
      console.error("No github_username in session metadata");
      return new Response("Missing metadata", { status: 400 });
    }

    const { error } = await supabase
      .from("users")
      .upsert({ github_username: username, analysis_authorized: true })
      .eq("github_username", username);

    if (error) {
      console.error("Supabase update failed:", error);
      return new Response("DB error", { status: 500 });
    }

    console.log(`Authorized analysis for ${username}`);
  }

  return new Response(JSON.stringify({ received: true }), {
    headers: { "Content-Type": "application/json" },
  });
});
