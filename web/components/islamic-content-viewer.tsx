'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronRight, BookOpen, FileText, Hash, Menu } from 'lucide-react';
import { HeadingsData, HeadingWithContent, getAllHeadingsWithContent } from '@/lib/data-loader';
import { HeadingsTableOfContents } from './headings-table-of-contents';
import Link from 'next/link';

interface IslamicContentViewerProps {
  data: HeadingsData;
  volumeId: number;
}

interface HeadingItemProps {
  heading: HeadingWithContent;
  volumeId: number;
}

function HeadingItem({ heading, volumeId }: HeadingItemProps) {
  const getHeadingIcon = (type: string) => {
    switch (type) {
      case 'heading_major':
        return <BookOpen className="w-5 h-5" />;
      case 'heading_minor':
        return <FileText className="w-4 h-4" />;
      default:
        return <Hash className="w-4 h-4" />;
    }
  };

  const getHeadingStyle = (type: string) => {
    switch (type) {
      case 'heading_major':
        return 'text-xl font-bold text-gray-900';
      case 'heading_minor':
        return 'text-lg font-semibold text-gray-800';
      default:
        return 'text-base font-medium text-gray-700';
    }
  };

  const getCardStyle = (type: string) => {
    switch (type) {
      case 'heading_major':
        return 'border-l-4 border-l-blue-500 bg-blue-50/50 hover:bg-blue-100/50';
      case 'heading_minor':
        return 'border-l-4 border-l-green-500 bg-green-50/50 hover:bg-green-100/50';
      default:
        return 'border-l-4 border-l-gray-400 bg-gray-50/50 hover:bg-gray-100/50';
    }
  };

  const CardContent = heading.content_count > 0 ? (
    <Link href={`/volume/${volumeId}/section/${heading.section_id}`}>
      <Card className={`mb-4 transition-all duration-200 cursor-pointer ${getCardStyle(heading.heading.type)}`}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3 flex-1">
              <div className="mt-1">
                {getHeadingIcon(heading.heading.type)}
              </div>
              <div className="flex-1">
                <CardTitle className={getHeadingStyle(heading.heading.type)}>
                  {heading.heading.text}
                </CardTitle>
                <div className="flex items-center gap-2 mt-2">
                  {heading.heading.page_number && (
                    <Badge variant="outline" className="text-xs">
                      Page {heading.heading.page_number}
                    </Badge>
                  )}
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  Click to view full content →
                </div>
              </div>
            </div>
            <ChevronRight className="w-5 h-5 text-gray-400 mt-1" />
          </div>
        </CardHeader>
      </Card>
    </Link>
  ) : (
    <Card className={`mb-4 opacity-60 ${getCardStyle(heading.heading.type)}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start gap-3">
          <div className="mt-1">
            {getHeadingIcon(heading.heading.type)}
          </div>
          <div className="flex-1">
            <CardTitle className={getHeadingStyle(heading.heading.type)}>
              {heading.heading.text}
            </CardTitle>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="outline" className="text-xs">
                No content
              </Badge>
              {heading.heading.page_number && (
                <Badge variant="outline" className="text-xs">
                  Page {heading.heading.page_number}
                </Badge>
              )}
            </div>
          </div>
        </div>
      </CardHeader>
    </Card>
  );

  return CardContent;
}

export function IslamicContentViewer({ data, volumeId }: IslamicContentViewerProps) {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set());
  const [filterType, setFilterType] = useState<string>('all');
  const [showTOC, setShowTOC] = useState<boolean>(true);

  const toggleSection = (sectionId: number) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId);
    } else {
      newExpanded.add(sectionId);
    }
    setExpandedSections(newExpanded);
  };

  const expandAll = () => {
    const allSectionIds = data.headings_to_content_mapping
      .filter(h => h.content_count > 0)
      .map(h => h.section_id);
    setExpandedSections(new Set(allSectionIds));
  };

  const collapseAll = () => {
    setExpandedSections(new Set());
  };

  const handleTOCHeadingClick = (sectionId: number) => {
    // Scroll to heading and expand it if it has content
    const element = document.getElementById(`heading-${sectionId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      
      // Auto-expand if it has content
      const heading = data.headings_to_content_mapping.find(h => h.section_id === sectionId);
      if (heading && heading.content_count > 0) {
        const newExpanded = new Set(expandedSections);
        newExpanded.add(sectionId);
        setExpandedSections(newExpanded);
      }
    }
  };

  const filteredHeadings = data.headings_to_content_mapping.filter(heading => {
    if (filterType === 'all') return true;
    return heading.heading.type === filterType;
  });

  const headingTypes = data.by_type ? Object.keys(data.by_type) : [];

  return (
    <div className="flex gap-6 max-w-7xl mx-auto px-4 py-8">
      {/* Table of Contents Sidebar */}
      {showTOC && (
        <div className="w-80 flex-shrink-0">
          <div className="sticky top-24">
            <HeadingsTableOfContents
              data={data}
              volumeId={volumeId}
              onHeadingClick={handleTOCHeadingClick}
            />
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 min-w-0">
        {/* Header with statistics */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-3xl font-bold text-gray-900">
              Volume {volumeId} - Headings & Content
            </h1>
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => setShowTOC(!showTOC)}
              >
                <Menu className="w-4 h-4 mr-2" />
                {showTOC ? 'Hide' : 'Show'} TOC
              </Button>
            </div>
          </div>

        </div>

        {/* Headings list */}
        <div className="space-y-4">
          {filteredHeadings.map((heading) => (
            <div key={heading.section_id} id={`heading-${heading.section_id}`}>
              <HeadingItem
                heading={heading}
                volumeId={volumeId}
              />
            </div>
          ))}
        </div>

        {filteredHeadings.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">
              No headings found for the selected filter.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
