import { useEffect, useState } from 'react';
import { API_KEY_CHANGED_EVENT, hasGeminiApiKey } from './geminiApiKey';

export function useStoredApiKey() {
  const [hasKey, setHasKey] = useState(hasGeminiApiKey);

  useEffect(() => {
    const sync = () => setHasKey(hasGeminiApiKey());

    window.addEventListener(API_KEY_CHANGED_EVENT, sync);
    window.addEventListener('storage', sync);

    return () => {
      window.removeEventListener(API_KEY_CHANGED_EVENT, sync);
      window.removeEventListener('storage', sync);
    };
  }, []);

  return hasKey;
}
