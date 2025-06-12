import { MetadataRoute } from 'next';
import { getAllHeadingsWithContent } from '@/lib/data-loader';

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const sitemapEntries: MetadataRoute.Sitemap = [];

  // Add static pages
  sitemapEntries.push({
    url: 'https://al-ifta.com',
    lastModified: new Date(),
    changeFrequency: 'daily',
    priority: 1.0,
  });

  // Generate sitemap entries for all volumes and their questions
  for (let volumeId = 1; volumeId <= 30; volumeId++) {
    try {
      const headings = await getAllHeadingsWithContent(volumeId);
      
      // Add volume page
      sitemapEntries.push({
        url: `https://al-ifta.com/volume/${volumeId}`,
        lastModified: new Date(),
        changeFrequency: 'weekly',
        priority: 0.8,
      });

      // Add question pages
      for (const heading of headings) {
        if (heading.content_count > 0) {
          sitemapEntries.push({
            url: `https://al-ifta.com/volume/${volumeId}/section/${heading.section_id}`,
            lastModified: new Date(),
            changeFrequency: 'weekly',
            priority: 0.6,
          });
        }
      }
    } catch (error) {
      // Skip volumes that don't exist
      continue;
    }
  }

  return sitemapEntries;
}