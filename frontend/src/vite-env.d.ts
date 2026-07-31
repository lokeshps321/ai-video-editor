/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CLERK_PUBLISHABLE_KEY: string;
  readonly VITE_API_BASE?: string;
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_REQUEST_TIMEOUT_MS?: string;
  readonly VITE_TIMELINE_CORE_V2?: string;
  readonly VITE_TIMELINE_TEST_HARNESS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
