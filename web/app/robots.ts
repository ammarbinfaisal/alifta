import { MetadataRoute } from 'next';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
    ],
    sitemap: 'https://www.al-ifta.com/sitemap.xml',
    host: 'https://www.al-ifta.com',
  };
}
