import { performanceMonitor } from './performance';

export interface ClassificationResult {
  material: string;
  confidence: number;
  recyclable: boolean;
  recommendations: string[];
}

const GEMINI_MODEL = import.meta.env.VITE_GEMINI_MODEL || 'gemini-3.1-flash-lite';
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

const CLASSIFICATION_PROMPT =
  "Analyze this image and identify the primary material of the waste item shown. Respond with the name of the material (e.g., 'Plastic', 'Cardboard', 'Glass'), and also mention what you have observed in the image.";

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1];
      if (!base64) {
        reject(new Error('Failed to encode image'));
        return;
      }
      resolve(base64);
    };
    reader.onerror = () => reject(new Error('Failed to read image data'));
    reader.readAsDataURL(blob);
  });
}

export async function classifyImage(
  imageElement: HTMLImageElement,
  apiKey: string
): Promise<ClassificationResult> {
  performanceMonitor.startTimer('classification');

  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Unable to process image');
    }

    canvas.width = imageElement.naturalWidth;
    canvas.height = imageElement.naturalHeight;
    ctx.drawImage(imageElement, 0, 0);

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((result) => {
        if (!result) {
          reject(new Error('Failed to convert image'));
          return;
        }
        resolve(result);
      }, 'image/jpeg', 0.8);
    });

    const base64Data = await blobToBase64(blob);

    const response = await fetch(`${GEMINI_API_URL}?key=${encodeURIComponent(apiKey)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { text: CLASSIFICATION_PROMPT },
              { inline_data: { mime_type: 'image/jpeg', data: base64Data } },
            ],
          },
        ],
      }),
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const message =
        payload?.error?.message ||
        (response.status === 401 || response.status === 403
          ? 'Invalid Gemini API key. Please check your key and try again.'
          : `Gemini API request failed (${response.status})`);
      throw new Error(message);
    }

    const material = payload?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
    if (!material) {
      throw new Error('No material identified');
    }

    const classificationTime = performanceMonitor.endTimer('classification');
    console.log(`Classification completed in ${classificationTime.toFixed(2)}ms`);

    const recyclabilityInfo = determineRecyclability(material.toLowerCase());

    return {
      material,
      confidence: 0.95,
      recyclable: recyclabilityInfo.recyclable,
      recommendations: recyclabilityInfo.recommendations,
    };
  } catch (error) {
    performanceMonitor.endTimer('classification');
    console.error('Classification error:', error);
    throw new Error(
      error instanceof Error ? error.message : 'Classification failed due to an unknown error'
    );
  }
}

function determineRecyclability(material: string): { recyclable: boolean; recommendations: string[] } {
  if (material.includes('plastic') || material.includes('bottle') || material.includes('container')) {
    return {
      recyclable: true,
      recommendations: [
        'Rinse and dry before recycling',
        'Remove caps and labels if possible',
        'Check local recycling guidelines for plastic type',
      ],
    };
  }

  if (material.includes('glass') || material.includes('jar')) {
    return {
      recyclable: true,
      recommendations: [
        'Rinse thoroughly to remove residue',
        'Remove metal lids and caps',
        'Separate by color if required locally',
      ],
    };
  }

  if (
    material.includes('paper') ||
    material.includes('cardboard') ||
    material.includes('magazine') ||
    material.includes('newspaper')
  ) {
    return {
      recyclable: true,
      recommendations: [
        'Keep dry and clean',
        'Remove plastic windows or tape',
        'Flatten boxes to save space',
      ],
    };
  }

  if (
    material.includes('metal') ||
    material.includes('aluminum') ||
    material.includes('steel') ||
    material.includes('can')
  ) {
    return {
      recyclable: true,
      recommendations: [
        'Rinse containers to remove food residue',
        'Remove non-metal parts like plastic lids',
        'Crush cans to save space',
      ],
    };
  }

  if (material.includes('food') || material.includes('organic') || material.includes('compost')) {
    return {
      recyclable: false,
      recommendations: [
        'Compost if possible',
        'Use for garden fertilizer',
        'Dispose in organic waste bin if available',
      ],
    };
  }

  return {
    recyclable: false,
    recommendations: [
      'Check local disposal guidelines',
      'Consider if item can be repurposed',
      'Dispose according to local regulations',
    ],
  };
}
