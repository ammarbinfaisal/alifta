import { MetadataRoute } from 'next'
import { glob } from 'glob'
import path from 'path'
import { getAllVolumes, getAllHeadingsWithContent } from '@/lib/data-loader'

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000'
  
  // Static pages
  const staticPages: MetadataRoute.Sitemap = [
    {
      url: baseUrl,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1,
    },
  ]

  // Discover volumes using glob
  const volsPath = path.join(process.cwd(), '..', 'vols')
  const volumeDirectories = await glob('vol*/', { cwd: volsPath })
  
  console.log(`Found ${volumeDirectories.length} volume directories:`, volumeDirectories)

  // Generate volume pages
  const volumePages: MetadataRoute.Sitemap = []
  
  try {
    const volumes = await getAllVolumes()
    
    for (const volume of volumes) {
      // Add main volume page
      volumePages.push({
        url: `${baseUrl}/volume/${volume.id}`,
        lastModified: new Date(volume.processed_date),
        changeFrequency: 'weekly',
        priority: 0.8,
      })

      // Add section pages for this volume
      try {
        const headingsWithContent = await getAllHeadingsWithContent(volume.id)
        
        for (const heading of headingsWithContent) {
          if (heading.content_count > 0) {
            volumePages.push({
              url: `${baseUrl}/volume/${volume.id}/section/${heading.section_id}`,
              lastModified: new Date(volume.processed_date),
              changeFrequency: 'monthly',
              priority: 0.6,
            })
          }
        }
        
        console.log(`Added ${headingsWithContent.filter(h => h.content_count > 0).length} section pages for volume ${volume.id}`)
      } catch (error) {
        console.warn(`Could not load sections for volume ${volume.id}:`, error)
      }
    }
  } catch (error) {
    console.error('Error generating volume sitemap entries:', error)
    
    // Fallback: use glob results to generate basic volume pages
    for (const volDir of volumeDirectories) {
      const match = volDir.match(/vol(\d+)/)
      if (match) {
        const volumeId = parseInt(match[1])
        volumePages.push({
          url: `${baseUrl}/volume/${volumeId}`,
          lastModified: new Date(),
          changeFrequency: 'weekly',
          priority: 0.8,
        })
      }
    }
  }

  console.log(`Generated sitemap with ${staticPages.length} static pages and ${volumePages.length} volume pages`)
  
  return [...staticPages, ...volumePages]
}

// Export additional function to get volume statistics
export async function getVolumeStats() {
  const volsPath = path.join(process.cwd(), '..', 'vols')
  const volumeDirectories = await glob('vol*/', { cwd: volsPath })
  
  const stats = {
    totalVolumes: volumeDirectories.length,
    volumeIds: [] as number[],
    totalSections: 0,
    totalSectionsWithContent: 0,
  }

  try {
    const volumes = await getAllVolumes()
    stats.volumeIds = volumes.map(v => v.id).sort((a, b) => a - b)
    
    for (const volume of volumes) {
      try {
        const headingsWithContent = await getAllHeadingsWithContent(volume.id)
        stats.totalSections += headingsWithContent.length
        stats.totalSectionsWithContent += headingsWithContent.filter(h => h.content_count > 0).length
      } catch (error) {
        console.warn(`Could not load sections for volume ${volume.id}:`, error)
      }
    }
  } catch (error) {
    console.error('Error getting volume stats:', error)
    // Fallback to glob results
    stats.volumeIds = volumeDirectories
      .map(dir => {
        const match = dir.match(/vol(\d+)/)
        return match ? parseInt(match[1]) : null
      })
      .filter((id): id is number => id !== null)
      .sort((a, b) => a - b)
  }

  return stats
}
