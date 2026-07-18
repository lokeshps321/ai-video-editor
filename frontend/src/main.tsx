import { lazy, StrictMode, Suspense, useLayoutEffect } from "react";
import { createRoot } from "react-dom/client";
import ErrorBoundary from "./ErrorBoundary";
import "./styles.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ClerkProvider, useAuth } from "@clerk/clerk-react";
import { setTokenGetter } from "./lib/api";
import ProtectedRoute from "./components/ProtectedRoute";

const App = lazy(() => import("./App"));
const LandingPage = lazy(() => import("./LandingPage"));
const SignInPage = lazy(() => import("./components/SignInPage"));
const SignUpPage = lazy(() => import("./components/SignUpPage"));

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY as
  | string
  | undefined;
if (!PUBLISHABLE_KEY) {
  throw new Error(
    "Missing VITE_CLERK_PUBLISHABLE_KEY. Create frontend/.env and add your Clerk publishable key.",
  );
}

// Syncs Clerk's getToken into the api module so every request gets a Bearer token.
function AuthTokenSync() {
  const { getToken, isLoaded, isSignedIn } = useAuth();

  // Register before App's regular effects can request protected API routes.
  // Otherwise the first project-list request can reach FastAPI without a
  // bearer token while Clerk is still hydrating its session.
  useLayoutEffect(() => {
    setTokenGetter(async () => {
      if (!isLoaded || !isSignedIn) return null;
      return getToken();
    });
    return () => setTokenGetter(null);
  }, [getToken, isLoaded, isSignedIn]);
  return null;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <ClerkProvider
        publishableKey={PUBLISHABLE_KEY}
        afterSignInUrl="/editor"
        afterSignUpUrl="/editor"
        afterSignOutUrl="/"
        signInUrl="/sign-in"
        signUpUrl="/sign-up"
      >
        <BrowserRouter>
          <AuthTokenSync />
          <Suspense fallback={<div className="routeLoading" />}>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/sign-in/*" element={<SignInPage />} />
              <Route path="/sign-up/*" element={<SignUpPage />} />
              <Route
                path="/editor"
                element={
                  <ProtectedRoute>
                    <App />
                  </ProtectedRoute>
                }
              />
              <Route path="*" element={<LandingPage />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </ClerkProvider>
    </ErrorBoundary>
  </StrictMode>,
);
