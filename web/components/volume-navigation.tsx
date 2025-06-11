import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ChevronLeft, ChevronRight, Home, Search } from 'lucide-react';

interface VolumeNavigationProps {
  currentVolumeId: number;
}

export function VolumeNavigation({ currentVolumeId }: VolumeNavigationProps) {
  const prevVolume = currentVolumeId > 1 ? currentVolumeId - 1 : null;
  const nextVolume = currentVolumeId < 30 ? currentVolumeId + 1 : null;

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-40 shadow-sm">
      <div className="max-w-6xl mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <Home className="w-4 h-4 mr-2" />
                All Volumes
              </Button>
            </Link>
            <Badge variant="outline">
              Volume {currentVolumeId.toString().padStart(2, '0')} of 30
            </Badge>
          </div>
          
          <div className="flex items-center gap-2">
            <Link href="/search">
              <Button variant="ghost" size="sm">
                <Search className="w-4 h-4" />
              </Button>
            </Link>
            
            {prevVolume && (
              <Link href={`/volume/${prevVolume}`}>
                <Button variant="outline" size="sm">
                  <ChevronLeft className="w-4 h-4 mr-1" />
                  Vol {prevVolume}
                </Button>
              </Link>
            )}
            
            {nextVolume && (
              <Link href={`/volume/${nextVolume}`}>
                <Button variant="outline" size="sm">
                  Vol {nextVolume}
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
