const STORAGE_KEY = 'gemini_api_key';
export const API_KEY_CHANGED_EVENT = 'gemini-api-key-changed';

export function getGeminiApiKey(): string | null {
  return localStorage.getItem(STORAGE_KEY);
}

export function setGeminiApiKey(key: string): void {
  localStorage.setItem(STORAGE_KEY, key.trim());
  window.dispatchEvent(new Event(API_KEY_CHANGED_EVENT));
}

export function clearGeminiApiKey(): void {
  localStorage.removeItem(STORAGE_KEY);
  window.dispatchEvent(new Event(API_KEY_CHANGED_EVENT));
}

export function hasGeminiApiKey(): boolean {
  return !!getGeminiApiKey();
}
