import { MetadataRoute } from 'next';
import { getAllVolumeIds } from '@/lib/data-loader';

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const baseUrl = 'https://al-ifta.com';
  const currentDate = new Date().toISOString();

  // Get all volume IDs
  const volumeIds = await getAllVolumeIds();

  // Base routes
  const routes: MetadataRoute.Sitemap = [
    {
      url: baseUrl,
      lastModified: currentDate,
      changeFrequency: 'weekly',
      priority: 1.0,
    },
  ];

  // Add volume pages
  for (const volumeId of volumeIds) {
    routes.push({
      url: `${baseUrl}/volume/${volumeId}`,
      lastModified: currentDate,
      changeFrequency: 'monthly',
      priority: 0.9,
    });
  }

  return routes;
}
