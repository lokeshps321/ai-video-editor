import { useAuth } from "@clerk/clerk-react";
import { Navigate } from "react-router-dom";
import type { ReactNode } from "react";

interface Props { children: ReactNode; }

export default function ProtectedRoute({ children }: Props) {
  const { isSignedIn, isLoaded } = useAuth();
  if (!isLoaded) {
    return <div style={{ minHeight: "100vh", background: "#0a0a0a" }} />;
  }
  if (!isSignedIn) {
    return <Navigate to="/sign-in" replace />;
  }
  return <>{children}</>;
}
