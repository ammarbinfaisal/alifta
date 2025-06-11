export interface HeadingWithContent {
  section_id: number;
  original_section_index: number;
  heading: {
    text: string;
    type: string;
    page_number?: number;
    confidence?: number;
  };
  content_items: {
    content_type: string;
    text: string;
    page_number?: number;
    confidence?: number;
  }[];
  content_count: number;
  content_types: string[];
  section_type?: string;
}

export async function getAllHeadingsWithContent(volumeId: number): Promise<HeadingWithContent[]> {
  try {
    const response = await fetch(`/api/volume/${volumeId}`);
    if (!response.ok) return [];
    const data = await response.json();
    return data.headings_to_content_mapping || [];
  } catch (error) {
    console.error('Error fetching headings:', error);
    return [];
  }
}
