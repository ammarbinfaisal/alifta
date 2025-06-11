#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');

/**
 * Convert complex Islamic content JSON to headings-to-content mapping format
 * Creates a mapping between headings and their associated content arrays
 * Usage: node convert-to-headings.js <input-file> <output-file>
 */

async function convertToHeadingsFormat(inputPath, outputPath) {
  try {
    console.log(`Reading input file: ${inputPath}`);
    const inputData = JSON.parse(await fs.readFile(inputPath, 'utf-8'));
    
    if (!inputData.sections || !Array.isArray(inputData.sections)) {
      throw new Error('Input file does not contain valid sections array');
    }

    const headings = [];
    let sectionId = 0;

    // Extract headings and create mapping to content
    const headingsToContent = [];
    
    inputData.sections.forEach((section, index) => {
      if (section.heading && section.heading.text) {
        const headingData = {
          section_id: sectionId++,
          original_section_index: index,
          heading: {
            text: section.heading.text,
            type: section.heading.content_type || 'heading_minor',
            page_number: section.heading.page_number,
            confidence: section.heading.confidence
          },
          content_items: section.content_items || [],
          content_count: (section.content_items || []).length,
          content_types: [...new Set((section.content_items || []).map(item => item.content_type))],
          section_type: section.type
        };
        
        headings.push({
          section_id: headingData.section_id,
          text: headingData.heading.text,
          type: headingData.heading.type
        });
        
        headingsToContent.push(headingData);
      }
    });

    // Group headings by type for easier access
    const byType = {};
    headings.forEach(heading => {
      if (!byType[heading.type]) {
        byType[heading.type] = [];
      }
      byType[heading.type].push(heading);
    });

    // Create content statistics
    const contentStats = {
      total_content_items: headingsToContent.reduce((sum, section) => sum + section.content_count, 0),
      content_types_found: [...new Set(headingsToContent.flatMap(section => section.content_types))],
      sections_with_content: headingsToContent.filter(section => section.content_count > 0).length,
      sections_without_content: headingsToContent.filter(section => section.content_count === 0).length
    };

    const outputData = {
      total_headings: headings.length,
      total_sections: headingsToContent.length,
      headings: headings,
      headings_to_content_mapping: headingsToContent,
      by_type: byType,
      content_statistics: contentStats,
      memory_enhanced: true
    };

    console.log(`Converting ${headings.length} headings with content mapping...`);
    console.log(`Types found: ${Object.keys(byType).join(', ')}`);
    console.log(`Total content items: ${contentStats.total_content_items}`);
    console.log(`Content types: ${contentStats.content_types_found.join(', ')}`);
    console.log(`Sections with content: ${contentStats.sections_with_content}/${headingsToContent.length}`);

    await fs.writeFile(outputPath, JSON.stringify(outputData, null, 2));
    console.log(`Successfully converted to: ${outputPath}`);

  } catch (error) {
    console.error('Error during conversion:', error.message);
    process.exit(1);
  }
}

async function convertDirectory(inputDir, outputDir) {
  try {
    await fs.mkdir(outputDir, { recursive: true });
    
    const files = await fs.readdir(inputDir);
    
    for (const file of files) {
      if (file.endsWith('.json') && file.includes('llm_parsed_complete')) {
        const inputPath = path.join(inputDir, file);
        const outputPath = path.join(outputDir, 'headings_index.json');
        
        console.log(`\nProcessing: ${inputPath}`);
        await convertToHeadingsFormat(inputPath, outputPath);
      }
    }
  } catch (error) {
    console.error('Error processing directory:', error.message);
    process.exit(1);
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage:');
    console.log('  node convert-to-headings.js <input-file> <output-file>');
    console.log('  node convert-to-headings.js --dir <input-dir> <output-dir>');
    console.log('');
    console.log('Examples:');
    console.log('  node convert-to-headings.js vols/vol1/llm_parsed_complete.json vols/vol1/headings_index.json');
    console.log('  node convert-to-headings.js --dir vols/vol1 vols/vol1');
    process.exit(1);
  }

  if (args[0] === '--dir') {
    if (args.length < 3) {
      console.error('Error: --dir requires input and output directory paths');
      process.exit(1);
    }
    await convertDirectory(args[1], args[2]);
  } else {
    if (args.length < 2) {
      console.error('Error: Please provide both input and output file paths');
      process.exit(1);
    }
    await convertToHeadingsFormat(args[0], args[1]);
  }
}

main().catch(console.error);
