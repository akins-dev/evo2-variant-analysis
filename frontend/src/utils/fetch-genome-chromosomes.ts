import type { ChromosomeFromSearch, UCSChromosomeResponse } from "@/types/genome-chromosomes";

export async function getGenomeChromosomes(genomeId: string) {
  const apiUrl = `https://api.genome.ucsc.edu/list/chromosomes?genome=${genomeId}`;
  const response = await fetch(apiUrl);
  if (!response.ok) {
    throw new Error("Failed to fetch chromosome list from UCSC API");
  }

  const chromosomeData = (await response.json()) as UCSChromosomeResponse;

  if (!chromosomeData.chromosomes) {
    throw new Error("UCSC API Error: missing chromosomes");
  }

  const chromosomes: ChromosomeFromSearch[] = [];

  for (const chromId in chromosomeData.chromosomes) {
    if (
      chromId.includes("_") ||
      chromId.includes("Un") ||
      chromId.includes("random")
    )
      continue;

    chromosomes.push({
      name: chromId,
      size: chromosomeData.chromosomes[chromId]!,
    });
  }

  // Sort chromosomes in natural order (1, 2, ..., 10, 11, ..., X, Y)
  chromosomes.sort((a, b) => {
    const numA = a.name.replace("chr", "");
    const numB = b.name.replace("chr", "");

    const isNumA = /^\d+$/.test(numA);
    const isNumB = /^\d+$/.test(numB);

    if (isNumA && isNumB) return Number(numA) - Number(numB);
    if (isNumA) return -1;
    if (isNumB) return 1;
    return numA.localeCompare(numB);
  });

  return { chromosomes };
}
