
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Book, FileText, Calendar } from 'lucide-react';
import { VolumeMetadata } from '@/lib/data-loader';

interface VolumeIndexProps {
  volumes: VolumeMetadata[];
}

export function VolumeIndex({ volumes }: VolumeIndexProps) {
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {volumes.map((volume) => (
          <Card key={volume.id} className="hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Book className="w-5 h-5 text-gray-600" />
                Volume {volume.id.toString().padStart(2, '0')}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <FileText className="w-4 h-4" />
                  {volume.total_headings} headings
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <Calendar className="w-4 h-4" />
                  {new Date(volume.processed_date).toLocaleDateString()}
                </div>
              </div>
              
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">{volume.total_headings} sections/Badge>
                <Badge variant="outline">{volume.title}</Badge>
              </div>
              
              <Link href={`/volume/${volume.id}`}>
                <Button className="w-full">
                  Read Volume {volume.id}
                </Button>
              </Link>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
