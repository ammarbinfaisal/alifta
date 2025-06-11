import fs from 'fs/promises';
import path from 'path';

export interface VolumeMetadata {
  id: number;
  title: string;
  total_headings: number;
  processed_date: string;
  file_path: string;
}

export interface Heading {
  section_id: number;
  text: string;
  type: string;
}

export interface ContentItem {
  content_type: string;
  text: string;
  page_number?: number;
  confidence?: number;
}

export interface HeadingWithContent {
  section_id: number;
  original_section_index: number;
  heading: {
    text: string;
    type: string;
    page_number?: number;
    confidence?: number;
  };
  content_items: ContentItem[];
  content_count: number;
  content_types: string[];
  section_type?: string;
}

export interface HeadingsData {
  total_headings: number;
  total_sections: number;
  headings: Heading[];
  headings_to_content_mapping: HeadingWithContent[];
  by_type?: {
    [key: string]: Heading[];
  };
  content_statistics?: {
    total_content_items: number;
    content_types_found: string[];
    sections_with_content: number;
    sections_without_content: number;
  };
  memory_enhanced?: boolean;
}

const DATA_DIR = path.join(process.cwd(), '..', 'vols');

export async function getAllVolumes(): Promise<VolumeMetadata[]> {
  try {
    const indexPath = path.join(process.cwd(), '..', 'data', 'index.json');
    const indexData = await fs.readFile(indexPath, 'utf-8');
    return JSON.parse(indexData).volumes;
  } catch (error) {
    // Fallback: generate from available volume directories
    const volumeDirs = await fs.readdir(DATA_DIR);
    const volumes: VolumeMetadata[] = [];
    
    for (const dir of volumeDirs) {
      const match = dir.match(/vol(\d+)/);
      if (match) {
        const id = parseInt(match[1]);
        const headingsPath = path.join(DATA_DIR, dir, 'headings_index.json');
        
        try {
          const data = await getVolumeData(id);
          if (data) {
            volumes.push({
              id,
              title: `Volume ${id}`, // Default title, can be enhanced
              total_headings: data.total_headings,
              processed_date: new Date().toISOString(),
              file_path: `${dir}/headings_index.json`
            });
          }
        } catch (err) {
          console.warn(`Could not load volume ${id}:`, err);
        }
      }
    }
    
    return volumes.sort((a, b) => a.id - b.id);
  }
}

export async function getAllVolumeIds(): Promise<number[]> {
  const volumes = await getAllVolumes();
  return volumes.map(v => v.id);
}

export async function getVolumeData(id: number): Promise<HeadingsData | null> {
  try {
    const filePath = path.join(DATA_DIR, `vol${id}`, 'headings_index.json');
    const data = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error(`Error loading volume ${id}:`, error);
    return null;
  }
}

export async function searchAcrossVolumes(query: string): Promise<SearchResult[]> {
  const volumes = await getAllVolumes();
  const results: SearchResult[] = [];
  
  for (const volume of volumes) {
    const data = await getVolumeData(volume.id);
    if (!data) continue;
    
    // Search in headings
    data.headings.forEach((heading, headingIndex) => {
      if (heading.text.toLowerCase().includes(query.toLowerCase())) {
        results.push({
          volumeId: volume.id,
          volumeTitle: volume.title,
          type: 'heading',
          headingIndex,
          text: heading.text,
          headingType: heading.type,
          sectionId: heading.section_id
        });
      }
    });
  }
  
  return results;
}

export async function getHeadingsByType(volumeId: number, type: string): Promise<Heading[]> {
  const data = await getVolumeData(volumeId);
  if (!data) return [];
  
  if (data.by_type && data.by_type[type]) {
    return data.by_type[type];
  }
  
  // Fallback: filter from main headings array
  return data.headings.filter(heading => heading.type === type);
}

export async function getHeadingById(volumeId: number, sectionId: number): Promise<Heading | null> {
  const data = await getVolumeData(volumeId);
  if (!data) return null;
  
  return data.headings.find(heading => heading.section_id === sectionId) || null;
}

export async function getHeadingWithContent(volumeId: number, sectionId: number): Promise<HeadingWithContent | null> {
  const data = await getVolumeData(volumeId);
  if (!data) return null;
  
  return data.headings_to_content_mapping.find(heading => heading.section_id === sectionId) || null;
}

export async function getAllHeadingsWithContent(volumeId: number): Promise<HeadingWithContent[]> {
  // Server-side usage only
  const data = await getVolumeData(volumeId);
  if (!data) return [];
  
  return data.headings_to_content_mapping;
}

interface SearchResult {
  volumeId: number;
  volumeTitle: string;
  type: 'heading';
  headingIndex: number;
  text: string;
  headingType: string;
  sectionId: number;
}
