export interface GenomeAssemblyFromSearch {
  id: string;
  name: string;
  sourceName: string;
  active: boolean;
}

interface UCSCGenome {
  description: string;
  nibPath: string;
  organism: string;
  defaultPos: string;
  active: number;
  orderKey: number;
  genome: string;
  scientificName: string;
  htmlPath: string;
  hgNearOk: number;
  hgPbOk: number;
  sourceName: string;
  taxId: number;
}

export interface UCSCGenomeResponse {
  downloadTime: string;
  downloadTimeStamp: number;
  dataTime: string;
  dataTimeStamp: number;
  ucscGenomes: Record<string, UCSCGenome>;
}
