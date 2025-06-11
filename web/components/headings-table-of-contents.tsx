'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { BookOpen, FileText, Hash, List, Grid } from 'lucide-react';
import { HeadingsData, Heading } from '@/lib/data-loader';

interface HeadingsTableOfContentsProps {
  data: HeadingsData;
  volumeId: number;
  onHeadingClick?: (sectionId: number) => void;
}

interface TOCItemProps {
  heading: Heading;
  onClick?: () => void;
}

function TOCItem({ heading, onClick }: TOCItemProps) {
  const getHeadingIcon = (type: string) => {
    switch (type) {
      case 'heading_major':
        return <BookOpen className="w-4 h-4" />;
      case 'heading_minor':
        return <FileText className="w-3 h-3" />;
      default:
        return <Hash className="w-3 h-3" />;
    }
  };

  const getItemStyle = (type: string) => {
    switch (type) {
      case 'heading_major':
        return 'font-semibold text-gray-900 py-2 px-3 border-l-2 border-blue-500 bg-blue-50/50';
      case 'heading_minor':
        return 'font-medium text-gray-800 py-1.5 px-3 ml-4 border-l-2 border-green-400 bg-green-50/30';
      default:
        return 'text-gray-700 py-1 px-3 ml-8 border-l border-gray-300 bg-gray-50/30';
    }
  };

  return (
    <div
      className={`${getItemStyle(heading.type)} cursor-pointer hover:bg-opacity-80 transition-colors rounded-r-md mb-1`}
      onClick={onClick}
    >
      <div className="flex items-center gap-2">
        {getHeadingIcon(heading.type)}
        <span className="text-sm line-clamp-2">{heading.text}</span>
        <Badge variant="outline" className="text-xs ml-auto">
          {heading.section_id}
        </Badge>
      </div>
    </div>
  );
}

export function HeadingsTableOfContents({ 
  data, 
  volumeId, 
  onHeadingClick 
}: HeadingsTableOfContentsProps) {
  const [viewMode, setViewMode] = useState<'list' | 'grouped'>('list');
  const [filterType, setFilterType] = useState<string>('all');

  const filteredHeadings = data.headings.filter(heading => {
    if (filterType === 'all') return true;
    return heading.type === filterType;
  });

  const groupedHeadings = data.by_type || {};
  const headingTypes = Object.keys(groupedHeadings);

  const handleHeadingClick = (sectionId: number) => {
    if (onHeadingClick) {
      onHeadingClick(sectionId);
    } else {
      // Scroll to heading in the main content
      const element = document.getElementById(`heading-${sectionId}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Table of Contents</CardTitle>
          <div className="flex gap-1">
            <Button
              variant={viewMode === 'list' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('list')}
            >
              <List className="w-4 h-4" />
            </Button>
            <Button
              variant={viewMode === 'grouped' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('grouped')}
            >
              <Grid className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        <div className="text-sm text-gray-600">
          Volume {volumeId} • {data.total_headings} sections
        </div>

        {/* Filter controls */}
        <div className="flex flex-wrap gap-1 mt-3">
          <Button
            variant={filterType === 'all' ? 'default' : 'outline'}
            size="sm"
            className="text-xs"
            onClick={() => setFilterType('all')}
          >
            All
          </Button>
        </div>
      </CardHeader>

      <CardContent className="pt-0 max-h-[calc(100vh-200px)] overflow-y-auto">
        {viewMode === 'list' ? (
          <div className="space-y-1">
            {filteredHeadings.map((heading) => (
              <TOCItem
                key={heading.section_id}
                heading={heading}
                onClick={() => handleHeadingClick(heading.section_id)}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {headingTypes.map(type => {
              const typeHeadings = groupedHeadings[type] || [];
              if (filterType !== 'all' && filterType !== type) return null;
              
              return (
                <div key={type}>
                  <div className="font-medium text-gray-900 mb-2 text-sm">
                    {type.replace('_', ' ')} ({typeHeadings.length})
                  </div>
                  <div className="space-y-1">
                    {typeHeadings.map((heading) => (
                      <TOCItem
                        key={heading.section_id}
                        heading={heading}
                        onClick={() => handleHeadingClick(heading.section_id)}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {filteredHeadings.length === 0 && (
          <div className="text-center py-8 text-gray-500 text-sm">
            No headings found for the selected filter.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
