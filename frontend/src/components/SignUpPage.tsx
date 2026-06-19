import { SignUp } from "@clerk/clerk-react";

export default function SignUpPage() {
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "#0a0a0a",
        gap: "24px",
      }}
    >
      <div style={{ textAlign: "center", marginBottom: "8px" }}>
        <h1
          style={{
            color: "#fff",
            fontSize: "1.8rem",
            fontWeight: 700,
            margin: 0,
          }}
        >
          ClipMind
        </h1>
        <p style={{ color: "#666", marginTop: "6px", fontSize: "0.95rem" }}>
          Create your account
        </p>
      </div>
      <SignUp
        routing="path"
        path="/sign-up"
        signInUrl="/sign-in"
        forceRedirectUrl="/editor"
      />
    </div>
  );
}
