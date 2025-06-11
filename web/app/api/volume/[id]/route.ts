import { NextRequest, NextResponse } from 'next/server';
import { getVolumeData } from '@/lib/data-loader';

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const volumeId = parseInt((await params).id);
    
    if (isNaN(volumeId) || volumeId < 1 || volumeId > 30) {
      return NextResponse.json(
        { error: 'Invalid volume ID' },
        { status: 400 }
      );
    }

    const volumeData = await getVolumeData(volumeId);
    
    if (!volumeData) {
      return NextResponse.json(
        { error: 'Volume not found' },
        { status: 404 }
      );
    }

    return NextResponse.json(volumeData);
  } catch (error) {
    console.error('Error fetching volume data:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
