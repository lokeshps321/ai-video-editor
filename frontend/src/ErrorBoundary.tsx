import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = {
  children: ReactNode;
};

type State = {
  hasError: boolean;
  errorMessage: string;
};

class ErrorBoundary extends Component<Props, State> {
  state: State = {
    hasError: false,
    errorMessage: ""
  };

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      errorMessage: error.message
    };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Keep this in console for debugging during local development.
    // eslint-disable-next-line no-console
    console.error("UI crash:", error, info);
  }

  render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          padding: "24px",
          fontFamily: "Sora, sans-serif",
          color: "#f8fdff"
        }}
      >
        <div
          style={{
            maxWidth: "760px",
            background: "rgba(10, 26, 35, 0.88)",
            border: "1px solid rgba(255,255,255,0.25)",
            borderRadius: "14px",
            padding: "18px"
          }}
        >
          <h2 style={{ marginTop: 0 }}>UI failed to render</h2>
          <p style={{ marginBottom: "10px", opacity: 0.9 }}>
            Open browser console and send this error to me so I can patch it immediately.
          </p>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              background: "rgba(255,255,255,0.08)",
              borderRadius: "10px",
              padding: "10px",
              margin: 0
            }}
          >
            {this.state.errorMessage || "Unknown render error"}
          </pre>
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;

