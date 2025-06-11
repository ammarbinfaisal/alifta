#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');

/**
 * Index.json Aggregator for Islamic Content Volumes
 * 
 * This script scans all volume JSON files and creates a comprehensive index
 * with metadata, statistics, and search information for the entire collection.
 */

class IslamicContentAggregator {
  constructor(volumesDir = './vols', outputDir = './data') {
    this.volumesDir = volumesDir;
    this.outputDir = outputDir;
    this.volumes = [];
    this.globalStats = {
      totalVolumes: 0,
      totalPages: 0,
      totalSections: 0,
      totalContentItems: 0,
      contentTypeDistribution: {},
      confidenceStats: {
        average: 0,
        min: 1,
        max: 0,
        distribution: {}
      },
      processingStats: {
        totalProcessingTime: 0,
        averageProcessingTimePerVolume: 0,
        totalMemoryPredictions: 0,
        successfulLLMCalls: 0,
        failedLLMCalls: 0
      },
      crossReferences: {
        quranReferences: {},
        hadithReferences: {},
        scholarNames: new Set(),
        topics: new Set()
      }
    };
  }

  /**
   * Main aggregation function
   */
  async aggregate() {
    console.log('🔄 Starting Islamic Content Aggregation...\n');
    
    try {
      // Ensure output directory exists
      await this.ensureDirectoryExists(this.outputDir);
      
      // Scan and process all volume files
      await this.scanVolumeFiles();
      
      // Generate comprehensive statistics
      await this.generateGlobalStatistics();
      
      // Create search index
      await this.createSearchIndex();
      
      // Generate the main index.json
      await this.generateIndexFile();
      
      // Generate additional metadata files
      await this.generateAdditionalFiles();
      
      console.log('✅ Aggregation completed successfully!\n');
      this.printSummary();
      
    } catch (error) {
      console.error('❌ Error during aggregation:', error);
      process.exit(1);
    }
  }

  /**
   * Scan the volumes directory for JSON files
   */
  async scanVolumeFiles() {
    console.log(`📁 Scanning volumes directory: ${this.volumesDir}`);
    
    try {
      const files = await fs.readdir(this.volumesDir, {
        withFileTypes: true
      });

      const volumeFiles = files
        .filter(file => file.name.match(/vol\d+/))
        .sort((a, b) => {
          const aNum = parseInt(a.name.match(/vol(\d+)/)[1]);
          const bNum = parseInt(b.name.match(/vol(\d+)/)[1]);
          return aNum - bNum;
        });

      console.log(`📚 Found ${volumeFiles.length} volume files`);

      for (const file of volumeFiles) {
        await this.processVolumeFile(`${file}/llm_parsed_complete.json`);
      }
      
    } catch (error) {
      throw new Error(`Failed to scan volumes directory: ${error.message}`);
    }
  }

  /**
   * Process individual volume file
   */
  async processVolumeFile(filename) {
    const filePath = path.join(this.volumesDir, filename);

    try {
      const data = await fs.readFile(filePath, 'utf-8');
      const volumeData = JSON.parse(data);
      
      // Extract volume metadata
      const volumeInfo = this.extractVolumeMetadata(volumeData, volumeId, filename);
      
      // Analyze content
      const contentAnalysis = this.analyzeVolumeContent(volumeData);
      
      // Combine metadata and analysis
      const completeVolumeInfo = {
        ...volumeInfo,
        ...contentAnalysis
      };
      
      this.volumes.push(completeVolumeInfo);
      
      // Update global statistics
      this.updateGlobalStats(volumeData, contentAnalysis);
      
    } catch (error) {
      console.error(`❌ Error processing ${filename}:`, error.message);
    }
  }

  /**
   * Extract basic metadata from volume
   */
  extractVolumeMetadata(volumeData, volumeId, filename) {
    const docInfo = volumeData.document_info || {};
    const processingStats = docInfo.processing_stats || {};
    const memoryStats = docInfo.memory_stats || {};

    return {
      id: volumeId,
      filename: filename,
      title: docInfo.title || `Volume ${volumeId}`,
      totalPages: docInfo.total_pages || 0,
      totalSections: docInfo.total_sections || volumeData.sections?.length || 0,
      processingMethod: docInfo.processing_method || 'unknown',
      processingDate: processingStats.end_time || processingStats.start_time || new Date().toISOString(),
      processingTime: processingStats.processing_time_seconds || 0,
      memoryPredictionsUsed: processingStats.memory_predictions_used || 0,
      successfulLLMCalls: processingStats.successful_llm_calls || 0,
      failedLLMCalls: processingStats.failed_llm_calls || 0,
      memoryStats: {
        patternsLearned: memoryStats.total_patterns_learned || 0,
        quranReferences: memoryStats.quran_references_tracked || 0,
        hadithReferences: memoryStats.hadith_references_tracked || 0,
        structureEntries: memoryStats.document_structure_entries || 0
      }
    };
  }

  /**
   * Analyze volume content in detail
   */
  analyzeVolumeContent(volumeData) {
    const sections = volumeData.sections || [];
    const analysis = {
      contentStats: {
        totalContentItems: 0,
        contentTypeDistribution: {},
        avgConfidenceByType: {},
        pageRange: { min: Infinity, max: -Infinity },
        sectionsWithHighConfidence: 0
      },
      topics: new Set(),
      headings: {
        major: [],
        minor: [],
        questions: []
      },
      crossReferences: {
        quranVerses: [],
        hadithReferences: [],
        citations: []
      },
      qualityMetrics: {
        averageConfidence: 0,
        lowConfidenceItems: 0,
        memoryAgreementRate: 0
      }
    };

    let totalConfidence = 0;
    let confidenceCount = 0;
    let memoryAgreements = 0;
    let memoryAgreementTotal = 0;

    sections.forEach(section => {
      // Process heading
      if (section.heading) {
        const heading = section.heading;
        const headingInfo = {
          text: heading.text,
          page: heading.page_number,
          confidence: heading.confidence
        };

        // Normalize heading_question to question for unified treatment
        let normalizedContentType = heading.content_type;
        if (normalizedContentType === 'heading_question') {
          normalizedContentType = 'question';
        }

        switch (normalizedContentType) {
          case 'heading_major':
            analysis.headings.major.push(headingInfo);
            break;
          case 'heading_minor':
            analysis.headings.minor.push(headingInfo);
            break;
          case 'question':
            analysis.headings.questions.push(headingInfo);
            break;
        }

        // Track confidence
        if (heading.confidence >= 0.8) {
          analysis.contentStats.sectionsWithHighConfidence++;
        }

        // Count heading as content item with normalized type
        analysis.contentStats.totalContentItems++;
        analysis.contentStats.contentTypeDistribution[normalizedContentType] = 
          (analysis.contentStats.contentTypeDistribution[normalizedContentType] || 0) + 1;
      }

      // Process content items
      section.content_items?.forEach(item => {
        analysis.contentStats.totalContentItems++;
        
        // Content type distribution - normalize heading_question to question
        let contentType = item.content_type;
        if (contentType === 'heading_question') {
          contentType = 'question';
        }
        analysis.contentStats.contentTypeDistribution[contentType] = 
          (analysis.contentStats.contentTypeDistribution[contentType] || 0) + 1;

        // Page range
        if (item.page_number) {
          analysis.contentStats.pageRange.min = Math.min(analysis.contentStats.pageRange.min, item.page_number);
          analysis.contentStats.pageRange.max = Math.max(analysis.contentStats.pageRange.max, item.page_number);
        }

        // Confidence tracking
        if (item.confidence !== undefined) {
          totalConfidence += item.confidence;
          confidenceCount++;
          
          if (item.confidence < 0.6) {
            analysis.qualityMetrics.lowConfidenceItems++;
          }
        }

        // Memory agreement tracking
        if (item.memory_agreement !== undefined) {
          memoryAgreementTotal++;
          if (item.memory_agreement) {
            memoryAgreements++;
          }
        }

        // Cross-reference detection
        this.detectCrossReferences(item, analysis.crossReferences);

        // Topic extraction
        this.extractTopics(item, analysis.topics);
      });
    });

    // Calculate averages
    analysis.qualityMetrics.averageConfidence = confidenceCount > 0 ? totalConfidence / confidenceCount : 0;
    analysis.qualityMetrics.memoryAgreementRate = memoryAgreementTotal > 0 ? memoryAgreements / memoryAgreementTotal : 0;

    // Fix page range
    if (analysis.contentStats.pageRange.min === Infinity) {
      analysis.contentStats.pageRange = { min: 0, max: 0 };
    }

    // Convert sets to arrays for JSON serialization
    analysis.topics = Array.from(analysis.topics);

    return analysis;
  }

  /**
   * Detect cross-references in content
   */
  detectCrossReferences(item, crossRefs) {
    const text = item.text.toLowerCase();

    // Quran verse patterns
    const quranPatterns = [
      /(\d+):(\d+)/g,  // Surah:Ayah
      /surah?\s+(\w+)/gi,
      /al-(\w+)/gi
    ];

    quranPatterns.forEach(pattern => {
      const matches = item.text.match(pattern);
      if (matches) {
        crossRefs.quranVerses.push(...matches.slice(0, 3)); // Limit to prevent spam
      }
    });

    // Hadith collection references
    const hadithCollections = ['bukhari', 'muslim', 'tirmidhi', 'abu dawud', 'nasa\'i', 'ibn majah'];
    hadithCollections.forEach(collection => {
      if (text.includes(collection)) {
        crossRefs.hadithReferences.push(collection);
      }
    });

    // Citation patterns
    if (item.content_type === 'citation' || text.includes('http') || text.includes('www.')) {
      crossRefs.citations.push(item.text.substring(0, 100)); // Truncate long citations
    }
  }

  /**
   * Extract topics from content
   */
  extractTopics(item, topics) {
    // Simple topic extraction based on content type and keywords
    // Normalize heading_question to question for unified treatment
    const normalizedContentType = item.content_type === 'heading_question' ? 'question' : item.content_type;
    
    if (normalizedContentType === 'heading_major' || normalizedContentType === 'heading_minor') {
      // Clean and add heading as topic
      const topic = item.text.replace(/^\(.*?\)/, '').trim(); // Remove page references
      if (topic.length > 3 && topic.length < 100) {
        topics.add(topic);
      }
    }

    // Extract common Islamic terms as topics
    const islamicTerms = [
      'prayer', 'salah', 'hajj', 'umrah', 'zakat', 'fasting', 'ramadan',
      'marriage', 'divorce', 'inheritance', 'business', 'trade', 'ribaa',
      'aqidah', 'tawhid', 'shirk', 'bid\'ah', 'sunnah', 'fiqh'
    ];

    const text = item.text.toLowerCase();
    islamicTerms.forEach(term => {
      if (text.includes(term)) {
        topics.add(term);
      }
    });
  }

  /**
   * Update global statistics
   */
  updateGlobalStats(volumeData, contentAnalysis) {
    const docInfo = volumeData.document_info || {};
    const processingStats = docInfo.processing_stats || {};

    // Basic counts
    this.globalStats.totalVolumes++;
    this.globalStats.totalPages += docInfo.total_pages || 0;
    this.globalStats.totalSections += docInfo.total_sections || 0;
    this.globalStats.totalContentItems += contentAnalysis.contentStats.totalContentItems;

    // Content type distribution
    Object.entries(contentAnalysis.contentStats.contentTypeDistribution).forEach(([type, count]) => {
      this.globalStats.contentTypeDistribution[type] = 
        (this.globalStats.contentTypeDistribution[type] || 0) + count;
    });

    // Processing stats
    this.globalStats.processingStats.totalProcessingTime += processingStats.processing_time_seconds || 0;
    this.globalStats.processingStats.totalMemoryPredictions += processingStats.memory_predictions_used || 0;
    this.globalStats.processingStats.successfulLLMCalls += processingStats.successful_llm_calls || 0;
    this.globalStats.processingStats.failedLLMCalls += processingStats.failed_llm_calls || 0;

    // Confidence stats
    const avgConf = contentAnalysis.qualityMetrics.averageConfidence;
    if (avgConf > 0) {
      this.globalStats.confidenceStats.min = Math.min(this.globalStats.confidenceStats.min, avgConf);
      this.globalStats.confidenceStats.max = Math.max(this.globalStats.confidenceStats.max, avgConf);
    }

    // Cross-references
    contentAnalysis.crossReferences.quranVerses.forEach(verse => {
      this.globalStats.crossReferences.quranReferences[verse] = 
        (this.globalStats.crossReferences.quranReferences[verse] || 0) + 1;
    });

    contentAnalysis.crossReferences.hadithReferences.forEach(hadith => {
      this.globalStats.crossReferences.hadithReferences[hadith] = 
        (this.globalStats.crossReferences.hadithReferences[hadith] || 0) + 1;
    });

    // Topics
    contentAnalysis.topics.forEach(topic => {
      this.globalStats.crossReferences.topics.add(topic);
    });
  }

  /**
   * Generate final global statistics
   */
  async generateGlobalStatistics() {
    console.log('📊 Generating global statistics...');

    // Calculate averages
    if (this.globalStats.totalVolumes > 0) {
      this.globalStats.processingStats.averageProcessingTimePerVolume = 
        this.globalStats.processingStats.totalProcessingTime / this.globalStats.totalVolumes;
    }

    // Calculate confidence average across all volumes
    const volumeConfidences = this.volumes
      .map(v => v.qualityMetrics.averageConfidence)
      .filter(c => c > 0);
    
    if (volumeConfidences.length > 0) {
      this.globalStats.confidenceStats.average = 
        volumeConfidences.reduce((a, b) => a + b, 0) / volumeConfidences.length;
    }

    // Convert sets to arrays
    this.globalStats.crossReferences.topics = Array.from(this.globalStats.crossReferences.topics);
    this.globalStats.crossReferences.scholarNames = Array.from(this.globalStats.crossReferences.scholarNames);
  }

  /**
   * Create search index for fast searching
   */
  async createSearchIndex() {
    console.log('🔍 Creating search index...');

    const searchIndex = {
      lastUpdated: new Date().toISOString(),
      totalEntries: 0,
      volumes: {},
      topics: {},
      contentTypes: {},
      headings: []
    };

    this.volumes.forEach(volume => {
      searchIndex.volumes[volume.id] = {
        title: volume.title,
        filename: volume.filename,
        pages: volume.totalPages,
        sections: volume.totalSections,
        majorHeadings: volume.headings.major.map(h => h.text),
        topics: volume.topics.slice(0, 20), // Limit topics
        contentTypes: Object.keys(volume.contentStats.contentTypeDistribution)
      };

      // Add major headings to global headings index
      volume.headings.major.forEach(heading => {
        searchIndex.headings.push({
          volumeId: volume.id,
          text: heading.text,
          page: heading.page,
          confidence: heading.confidence
        });
      });

      // Index topics
      volume.topics.forEach(topic => {
        if (!searchIndex.topics[topic]) {
          searchIndex.topics[topic] = [];
        }
        searchIndex.topics[topic].push(volume.id);
      });

      searchIndex.totalEntries++;
    });

    // Write search index
    const searchIndexPath = path.join(this.outputDir, 'search-index.json');
    await fs.writeFile(searchIndexPath, JSON.stringify(searchIndex, null, 2));
    console.log(`   ✅ Search index saved: ${searchIndexPath}`);
  }

  /**
   * Generate the main index.json file
   */
  async generateIndexFile() {
    console.log('📝 Generating main index.json...');

    const indexData = {
      metadata: {
        generated: new Date().toISOString(),
        generator: 'Islamic Content Aggregator v1.0',
        schema_version: '1.0.0',
        collection_title: 'Majmoo\'al-Fatawa of Ibn Bazz'
      },
      collection: {
        title: 'Majmoo\'al-Fatawa of Ibn Bazz',
        description: 'Complete collection of Fatwas by Sheikh Abdul Aziz ibn Baz',
        author: 'Sheikh Abdul Aziz ibn Abdullah ibn Baz',
        totalVolumes: this.globalStats.totalVolumes,
        totalPages: this.globalStats.totalPages,
        totalSections: this.globalStats.totalSections,
        totalContentItems: this.globalStats.totalContentItems,
        processingCompleted: new Date().toISOString()
      },
      statistics: this.globalStats,
      volumes: this.volumes.map(volume => ({
        // Essential metadata for the index
        id: volume.id,
        filename: volume.filename,
        title: volume.title,
        totalPages: volume.totalPages,
        totalSections: volume.totalSections,
        processingDate: volume.processingDate,
        processingTime: volume.processingTime,
        
        // Content overview
        contentItems: volume.contentStats.totalContentItems,
        pageRange: volume.contentStats.pageRange,
        majorHeadings: volume.headings.major.length,
        topics: volume.topics.slice(0, 10), // Top 10 topics
        
        // Quality metrics
        averageConfidence: Math.round(volume.qualityMetrics.averageConfidence * 100) / 100,
        memoryAgreementRate: Math.round(volume.qualityMetrics.memoryAgreementRate * 100) / 100,
        
        // Processing info
        memoryPredictionsUsed: volume.memoryPredictionsUsed,
        successfulLLMCalls: volume.successfulLLMCalls,
        
        // Most common content types (top 5)
        topContentTypes: Object.entries(volume.contentStats.contentTypeDistribution)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 5)
          .map(([type, count]) => ({ type, count }))
      }))
    };

    const indexPath = path.join(this.outputDir, 'index.json');
    await fs.writeFile(indexPath, JSON.stringify(indexData, null, 2));
    console.log(`   ✅ Main index saved: ${indexPath}`);
  }

  /**
   * Generate additional metadata files
   */
  async generateAdditionalFiles() {
    console.log('📄 Generating additional metadata files...');

    // Topic index
    const topicsIndex = {
      lastUpdated: new Date().toISOString(),
      totalTopics: this.globalStats.crossReferences.topics.length,
      topics: this.globalStats.crossReferences.topics.sort(),
      topicsByVolume: {}
    };

    this.volumes.forEach(volume => {
      topicsIndex.topicsByVolume[volume.id] = volume.topics;
    });

    await fs.writeFile(
      path.join(this.outputDir, 'topics-index.json'),
      JSON.stringify(topicsIndex, null, 2)
    );

    // Cross-references index
    const crossRefsIndex = {
      lastUpdated: new Date().toISOString(),
      quranReferences: this.globalStats.crossReferences.quranReferences,
      hadithReferences: this.globalStats.crossReferences.hadithReferences,
      topReferences: {
        quran: Object.entries(this.globalStats.crossReferences.quranReferences)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 20),
        hadith: Object.entries(this.globalStats.crossReferences.hadithReferences)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 10)
      }
    };

    await fs.writeFile(
      path.join(this.outputDir, 'cross-references.json'),
      JSON.stringify(crossRefsIndex, null, 2)
    );

    // Processing report
    const processingReport = {
      generated: new Date().toISOString(),
      summary: {
        totalVolumes: this.globalStats.totalVolumes,
        successRate: `${((this.globalStats.processingStats.successfulLLMCalls / 
          Math.max(1, this.globalStats.processingStats.successfulLLMCalls + this.globalStats.processingStats.failedLLMCalls)) * 100).toFixed(1)}%`,
        totalProcessingTime: `${(this.globalStats.processingStats.totalProcessingTime / 3600).toFixed(2)} hours`,
        averageTimePerVolume: `${(this.globalStats.processingStats.averageProcessingTimePerVolume / 60).toFixed(1)} minutes`
      },
      qualityMetrics: {
        averageConfidence: `${(this.globalStats.confidenceStats.average * 100).toFixed(1)}%`,
        totalMemoryPredictions: this.globalStats.processingStats.totalMemoryPredictions,
        contentTypesCovered: Object.keys(this.globalStats.contentTypeDistribution).length
      },
      volumeDetails: this.volumes.map(v => ({
        id: v.id,
        title: v.title,
        processingTime: `${(v.processingTime / 60).toFixed(1)} min`,
        confidence: `${(v.qualityMetrics.averageConfidence * 100).toFixed(1)}%`,
        memoryAgreement: `${(v.qualityMetrics.memoryAgreementRate * 100).toFixed(1)}%`,
        contentItems: v.contentStats.totalContentItems
      }))
    };

    await fs.writeFile(
      path.join(this.outputDir, 'processing-report.json'),
      JSON.stringify(processingReport, null, 2)
    );

    console.log('   ✅ Additional files generated');
  }

  /**
   * Ensure directory exists
   */
  async ensureDirectoryExists(dir) {
    try {
      await fs.access(dir);
    } catch {
      await fs.mkdir(dir, { recursive: true });
    }
  }

  /**
   * Print summary of aggregation
   */
  printSummary() {
    console.log('📋 AGGREGATION SUMMARY');
    console.log('=====================================');
    console.log(`📚 Total Volumes: ${this.globalStats.totalVolumes}`);
    console.log(`📄 Total Pages: ${this.globalStats.totalPages.toLocaleString()}`);
    console.log(`📑 Total Sections: ${this.globalStats.totalSections.toLocaleString()}`);
    console.log(`🔤 Total Content Items: ${this.globalStats.totalContentItems.toLocaleString()}`);
    console.log(`⏱️  Total Processing Time: ${(this.globalStats.processingStats.totalProcessingTime / 3600).toFixed(2)} hours`);
    console.log(`🎯 Average Confidence: ${(this.globalStats.confidenceStats.average * 100).toFixed(1)}%`);
    console.log(`🧠 Memory Predictions: ${this.globalStats.processingStats.totalMemoryPredictions.toLocaleString()}`);
    console.log(`🔍 Topics Found: ${this.globalStats.crossReferences.topics.length}`);
    console.log(`📖 Quran References: ${Object.keys(this.globalStats.crossReferences.quranReferences).length}`);
    console.log(`📚 Hadith References: ${Object.keys(this.globalStats.crossReferences.hadithReferences).length}`);
    console.log('=====================================');
    console.log('✅ Files generated:');
    console.log('   📄 index.json - Main collection index');
    console.log('   🔍 search-index.json - Search functionality');
    console.log('   🏷️  topics-index.json - Topic categorization');
    console.log('   🔗 cross-references.json - Quran & Hadith refs');
    console.log('   📊 processing-report.json - Quality metrics');
  }
}

// CLI Usage
async function main() {
  const args = process.argv.slice(2);
  const volumesDir = args[0] || './vols';
  const outputDir = args[1] || './data';

  console.log('🕌 Islamic Content Index Aggregator');
  console.log('=====================================');
  console.log(`📁 Volumes Directory: ${volumesDir}`);
  console.log(`📁 Output Directory: ${outputDir}`);
  console.log('');

  const aggregator = new IslamicContentAggregator(volumesDir, outputDir);
  await aggregator.aggregate();
}

main().then(console.log).catch(console.log)
