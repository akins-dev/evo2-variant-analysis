export interface UCSChromosomeResponse {
  downloadTime: string;
  downloadTimeStamp: number;
  genome: string;
  dataTime: string;
  dataTimeStamp: number;
  chromCount: number;
  chromosomes: Record<string, number>;
}

export interface ChromosomeFromSearch {
  name: string;
  size: number;
}
