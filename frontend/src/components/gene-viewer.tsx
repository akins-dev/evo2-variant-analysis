"use client";

import { Button } from "./ui/button";
import { ArrowLeft } from "lucide-react";
import { useEffect, useState } from "react";
import type { GeneBounds, GeneRecord } from "@/types/gene-details";
import { fetchGeneDetails } from "@/utils/fetch-gene-details";
import type { GeneFromSearch } from "@/types/gene-search";

const GeneViewer = ({
  gene,
  genomeId,
  onClose,
}: {
  gene: GeneFromSearch;
  genomeId: string;
  onClose: () => void;
}) => {
  const [geneBounds, setGeneBounds] = useState<GeneBounds | null>(null);
  const [geneDetail, setGeneDetail] = useState<GeneRecord | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [startPostion, setStartPosition] = useState<string>("");
  const [endPosition, setEndPosition] = useState<string>("");

  useEffect(() => {
    const initializeGeneData = async () => {
      setIsLoading(true);
      setError(null);
      setGeneDetail(null);
      setStartPosition("");
      setEndPosition("");

      if (!gene.geneId) {
        setError("Gene ID is missing, cannot fetch details.");
        setIsLoading(false);
        return;
      }

      try {
        const { geneDetails, initialRange, geneBounds } =
          await fetchGeneDetails(gene.geneId);
        setGeneDetail(geneDetails);
        setGeneBounds(geneBounds);

        if (initialRange) {
          setStartPosition(initialRange.start.toString());
          setEndPosition(initialRange.end.toString());

          // fetch gene sequence
        }
      } catch (error) {
        console.log("fetchGeneDetails error: ", error);
        setError("Failed to fetch gene details. Please try again.");
      } finally {
        setIsLoading(false);
      }
    };
  }, [gene, genomeId]);
  return (
    <div className="space-y-6">
      <Button
        variant="ghost"
        className="text-primary hover:bg-secondary/70 cursor-pointer"
        size="sm"
        onClick={onClose}
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to results
      </Button>
    </div>
  );
};

export default GeneViewer;
