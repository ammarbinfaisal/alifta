import { MetadataRoute } from 'next';

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'Alifta - Islamic Fatwa Collection',
    short_name: 'Alifta',
    description: 'Comprehensive Islamic fatwa collection providing authentic religious guidance and jurisprudence rulings from qualified Islamic scholars.',
    start_url: '/',
    display: 'standalone',
    background_color: '#ffffff',
    theme_color: '#2b5797',
    orientation: 'portrait',
    categories: ['education', 'reference', 'books'],
    lang: 'en',
    dir: 'ltr',
    icons: [
      {
        src: '/icon-192x192.png',
        sizes: '192x192',
        type: 'image/png',
        purpose: 'maskable'
      },
      {
        src: '/icon-512x512.png',
        sizes: '512x512',
        type: 'image/png',
        purpose: 'maskable'
      },
      {
        src: '/apple-touch-icon.png',
        sizes: '180x180',
        type: 'image/png'
      }
    ],
    screenshots: [
      {
        src: '/screenshot-wide.png',
        sizes: '1280x720',
        type: 'image/png',
        form_factor: 'wide'
      },
      {
        src: '/screenshot-narrow.png',
        sizes: '750x1334',
        type: 'image/png',
        form_factor: 'narrow'
      }
    ]
  };
}
