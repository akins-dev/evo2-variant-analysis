import type { GeneBounds, GeneRecord, GeneSummaryResponse } from "@/types/gene-details";

export async function fetchGeneDetails(geneId: string): Promise<{
  geneDetails: GeneRecord | null;
  geneBounds: GeneBounds | null;
  initialRange: { start: number; end: number } | null;
}> {
  try {
    const detailUrl = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id=${geneId}&retmode=json`;
    const response = await fetch(detailUrl);

    if (!response.ok) {
      console.log(
        "fetchGeneDetails[ERROR]: Failed to fetch gene details =>",
        response.statusText,
      );
      return {
        geneDetails: null,
        geneBounds: null,
        initialRange: null,
      };
    }
    const data = (await response.json()) as GeneSummaryResponse;

    if (data.result?.[geneId]) {
      const detail = data.result[geneId] as GeneRecord;

      if (detail.genomicinfo && detail.genomicinfo.length > 0) {
        const genomicInfo = detail.genomicinfo[0]!;

        // positions
        const minPos = Math.min(genomicInfo.chrstart, genomicInfo.chrstop);
        const maxPos = Math.max(genomicInfo.chrstart, genomicInfo.chrstop);
        const bounds = { min: minPos, max: maxPos };

        // initial range (first 10,000 or full gene if smaller)
        const geneSize = maxPos - minPos;
        const seqStart = minPos;
        const seqEnd = geneSize > 10000 ? minPos + 10000 : maxPos;
        const range = { start: seqStart, end: seqEnd };

        return {
          geneDetails: detail,
          geneBounds: bounds,
          initialRange: range,
        };
      }
    }
    return {
      geneDetails: null,
      geneBounds: null,
      initialRange: null,
    };
  } catch (error) {
    console.log("fetchGeneDetails[ERROR]:", error);
    return {
      geneDetails: null,
      geneBounds: null,
      initialRange: null,
    };
  }
}
