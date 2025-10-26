export interface GeneBounds {
  min: number;
  max: number;
}

export interface GeneSummaryResponse {
  header: {
    type: string;
    version: string;
  };
  result: {
    uids: string[];
    [uid: string]: GeneRecord | string[]; // "uids" is a string[], the rest are GeneRecord objects
  };
}

export interface GeneRecord {
  uid: string;
  name: string;
  description: string;
  status: string;
  currentid: string;
  chromosome: string;
  geneticsource: string;
  maplocation: string;
  otheraliases: string;
  otherdesignations: string;
  nomenclaturesymbol: string;
  nomenclaturename: string;
  nomenclaturestatus: string;
  mim: string[];
  genomicinfo: GenomicInfo[];
  geneweight: number;
  summary: string;
  chrsort: string;
  chrstart: number;
  organism: Organism;
  locationhist: LocationHistory[];
}

export interface GenomicInfo {
  chrloc: string;
  chraccver: string;
  chrstart: number;
  chrstop: number;
  exoncount: string;
}

export interface Organism {
  scientificname: string;
  commonname: string;
  taxid: number;
}

export interface LocationHistory {
  annotationrelease: string;
  assemblyaccver: string;
  chraccver: string;
  chrstart: number;
  chrstop: number;
}
