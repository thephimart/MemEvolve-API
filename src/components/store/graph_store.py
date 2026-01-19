"""
Graph Database Storage Backend for MemEvolve

Implements graph-based storage using Neo4j for memory units with relationship support.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging

from .base import StorageBackend


class GraphStorageBackend(StorageBackend):
    """Neo4j-based graph storage backend for memory units with relationship support."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        create_relationships: bool = True,
        embedding_function: Optional[callable] = None
    ):
        """
        Initialize Neo4j graph storage backend.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            create_relationships: Whether to create relationships between similar memories
            embedding_function: Function to compute embeddings for similarity relationships
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.create_relationships = create_relationships
        self.embedding_function = embedding_function
        self.driver = None
        self.logger = logging.getLogger(__name__)
        self._connect()

    def _connect(self):
        """Connect to Neo4j database."""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                database=self.database
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info(f"Connected to Neo4j at {self.uri}")
        except ImportError:
            self.logger.warning("Neo4j driver not available. Using fallback in-memory graph.")
            self._setup_fallback_graph()
        except Exception as e:
            self.logger.warning(f"Neo4j connection failed: {e}. Using fallback in-memory graph.")
            self._setup_fallback_graph()

    def _setup_fallback_graph(self):
        """Set up NetworkX-based fallback graph for development/testing."""
        try:
            import networkx as nx
            self.graph = nx.DiGraph()
            self.node_data = {}
            self.logger.info("Using NetworkX fallback graph storage")
        except ImportError:
            self.logger.warning("NetworkX not available. Installing basic fallback.")
            # Basic fallback without NetworkX - just use dict storage
            self.graph = None
            self.node_data = {}

    def _get_node_id(self, unit: Dict[str, Any]) -> str:
        """Generate a unique node ID for a memory unit."""
        # Use content hash for deduplication
        content = json.dumps(unit, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def store(self, unit: Dict[str, Any]) -> str:
        """Store a memory unit as a graph node."""
        unit_id = self._get_node_id(unit)

        if self.driver:
            # Neo4j implementation
            return self._store_neo4j(unit, unit_id)
        else:
            # NetworkX fallback
            return self._store_networkx(unit, unit_id)

    def _store_neo4j(self, unit: Dict[str, Any], unit_id: str) -> str:
        """Store unit in Neo4j."""
        with self.driver.session() as session:
            # Create or update node
            query = """
            MERGE (m:Memory {id: $id})
            SET m.content = $content,
                m.type = $type,
                m.tags = $tags,
                m.metadata = $metadata,
                m.created_at = $created_at,
                m.updated_at = $updated_at
            RETURN m.id
            """

            metadata = unit.get("metadata", {})
            tags = unit.get("tags", [])

            result = session.run(
                query,
                id=unit_id,
                content=unit.get("content", ""),
                type=unit.get("type", "unknown"),
                tags=tags,
                metadata=json.dumps(metadata),
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat()
            )
            result.consume()  # Ensure query execution

            # Create relationships if enabled
            if self.create_relationships:
                self._create_relationships_neo4j(session, unit_id, unit)

            return unit_id

    def _store_networkx(self, unit: Dict[str, Any], unit_id: str) -> str:
        """Store unit in NetworkX fallback graph or basic dict storage."""
        # Store node data
        self.node_data[unit_id] = {
            **unit,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        # Add node to graph if NetworkX is available
        if self.graph is not None:
            self.graph.add_node(unit_id, **self.node_data[unit_id])

            # Create relationships if enabled
            if self.create_relationships:
                self._create_relationships_networkx(unit_id, unit)

        return unit_id

    def _create_relationships_neo4j(self, session, unit_id: str, unit: Dict[str, Any]):
        """Create relationships between memories in Neo4j."""
        # Create similarity relationships based on type and tags
        unit_type = unit.get("type", "unknown")
        unit_tags = unit.get("tags", [])

        # Find similar memories and create relationships
        similarity_query = """
        MATCH (m:Memory)
        WHERE m.id <> $id AND m.type = $type
        WITH m, size([tag IN m.tags WHERE tag IN $tags]) as common_tags
        WHERE common_tags > 0
        MERGE (m)-[r:SIMILAR_TO {weight: common_tags}]->(target:Memory {id: $id})
        """

        session.run(
            similarity_query,
            id=unit_id,
            type=unit_type,
            tags=unit_tags
        )

    def _create_relationships_networkx(self, unit_id: str, unit: Dict[str, Any]):
        """Create relationships between memories in NetworkX graph."""
        if not self.graph:
            return

        unit_type = unit.get("type", "unknown")
        unit_tags = set(unit.get("tags", []))

        # Find similar existing nodes and create edges
        for existing_id, existing_data in self.node_data.items():
            if existing_id == unit_id:
                continue

            existing_type = existing_data.get("type", "unknown")
            existing_tags = set(existing_data.get("tags", []))

            # Create relationship if same type and shared tags
            if existing_type == unit_type and unit_tags & existing_tags:
                weight = len(unit_tags & existing_tags)
                self.graph.add_edge(existing_id, unit_id, weight=weight, type="similar")

    def store_batch(self, units: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memory units efficiently."""
        unit_ids = []

        if self.driver:
            # Batch Neo4j operations
            for unit in units:
                unit_id = self._get_node_id(unit)
                self._store_neo4j(unit, unit_id)
                unit_ids.append(unit_id)
        else:
            # Batch NetworkX operations
            for unit in units:
                unit_id = self._get_node_id(unit)
                self._store_networkx(unit, unit_id)
                unit_ids.append(unit_id)

        return unit_ids

    def retrieve(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory unit by ID."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) RETURN m",
                    id=unit_id
                )
                record = result.single()
                if record:
                    node_data = dict(record["m"])
                    # Parse metadata back to dict
                    if "metadata" in node_data and isinstance(node_data["metadata"], str):
                        try:
                            node_data["metadata"] = json.loads(node_data["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    return node_data
        else:
            # NetworkX retrieval
            return self.node_data.get(unit_id)

        return None

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieve all stored memory units."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run("MATCH (m:Memory) RETURN m")
                units = []
                for record in result:
                    node_data = dict(record["m"])
                    # Parse metadata back to dict
                    if "metadata" in node_data and isinstance(node_data["metadata"], str):
                        try:
                            node_data["metadata"] = json.loads(node_data["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    units.append(node_data)
                return units
        else:
            # NetworkX retrieval
            return list(self.node_data.values())

    def update(self, unit_id: str, unit: Dict[str, Any]) -> bool:
        """Update a memory unit by ID."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (m:Memory {id: $id})
                    SET m.content = $content,
                        m.type = $type,
                        m.tags = $tags,
                        m.metadata = $metadata,
                        m.updated_at = $updated_at
                    RETURN count(m) > 0 as updated
                    """,
                    id=unit_id,
                    content=unit.get("content", ""),
                    type=unit.get("type", "unknown"),
                    tags=unit.get("tags", []),
                    metadata=json.dumps(unit.get("metadata", {})),
                    updated_at=datetime.now(timezone.utc).isoformat()
                )
                record = result.single()
                return record["updated"] if record else False
        else:
            # NetworkX update
            if unit_id in self.node_data:
                self.node_data[unit_id].update(unit)
                self.node_data[unit_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
                return True
            return False

    def delete(self, unit_id: str) -> bool:
        """Delete a memory unit by ID."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) DELETE m RETURN count(m) > 0 as deleted",
                    id=unit_id
                )
                record = result.single()
                return record["deleted"] if record else False
        else:
            # NetworkX delete
            if unit_id in self.node_data:
                del self.node_data[unit_id]
                if self.graph:
                    self.graph.remove_node(unit_id)
                return True
            return False

    def exists(self, unit_id: str) -> bool:
        """Check if a memory unit exists."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) RETURN count(m) > 0 as exists",
                    id=unit_id
                )
                record = result.single()
                return record["exists"] if record else False
        else:
            return unit_id in self.node_data

    def count(self) -> int:
        """Get the count of stored memory units."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run("MATCH (m:Memory) RETURN count(m) as count")
                record = result.single()
                return record["count"] if record else 0
        else:
            return len(self.node_data)

    def clear(self) -> None:
        """Clear all stored memory units."""
        if self.driver:
            with self.driver.session() as session:
                session.run("MATCH (m:Memory) DELETE m")
        else:
            self.node_data.clear()
            if self.graph:
                self.graph.clear()

    def query_related(self, unit_id: str, relationship_type: str = "SIMILAR_TO",
                     max_depth: int = 2, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query related memory units through graph relationships.

        Args:
            unit_id: Starting memory unit ID
            relationship_type: Type of relationship to traverse
            max_depth: Maximum traversal depth
            limit: Maximum number of results to return

        Returns:
            List of related memory units with relationship info
        """
        if self.driver:
            return self._query_related_neo4j(unit_id, relationship_type, max_depth, limit)
        else:
            return self._query_related_networkx(unit_id, relationship_type, max_depth, limit)

    def _query_related_neo4j(self, unit_id: str, relationship_type: str,
                           max_depth: int, limit: int) -> List[Dict[str, Any]]:
        """Query related units in Neo4j."""
        with self.driver.session() as session:
            query = f"""
            MATCH (start:Memory {{id: $id}})-[r:{relationship_type}*1..{max_depth}]-(related:Memory)
            WHERE related.id <> $id
            RETURN related, r, length(r) as depth
            ORDER BY length(r), r.weight DESC
            LIMIT $limit
            """

            result = session.run(query, id=unit_id, limit=limit)
            related_units = []

            for record in result:
                node_data = dict(record["related"])
                # Parse metadata
                if "metadata" in node_data and isinstance(node_data["metadata"], str):
                    try:
                        node_data["metadata"] = json.loads(node_data["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                related_units.append({
                    "unit": node_data,
                    "relationship": dict(record["r"][-1]) if record["r"] else {},
                    "depth": record["depth"]
                })

            return related_units

    def _query_related_networkx(self, unit_id: str, relationship_type: str,
                              max_depth: int, limit: int) -> List[Dict[str, Any]]:
        """Query related units in NetworkX graph or fallback to basic similarity."""
        if not self.graph:
            # Fallback: find similar units by tags when NetworkX not available
            return self._query_related_fallback(unit_id, relationship_type, max_depth, limit)

        if unit_id not in self.graph:
            return []

        related_units = []

        # Simple BFS to find related nodes
        visited = set([unit_id])
        queue = [(unit_id, 0)]  # (node_id, depth)

        while queue and len(related_units) < limit:
            current_id, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            # Get neighbors
            for neighbor_id in self.graph.neighbors(current_id):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    edge_data = self.graph.get_edge_data(current_id, neighbor_id, {})

                    if edge_data.get("type") == relationship_type.replace("_TO", "").lower():
                        related_units.append({
                            "unit": self.node_data.get(neighbor_id, {}),
                            "relationship": edge_data,
                            "depth": depth + 1
                        })

                        if len(related_units) >= limit:
                            break

                    queue.append((neighbor_id, depth + 1))

        return related_units

    def _query_related_fallback(self, unit_id: str, relationship_type: str,
                              max_depth: int, limit: int) -> List[Dict[str, Any]]:
        """Fallback query when NetworkX is not available - find by tag similarity."""
        if unit_id not in self.node_data:
            return []

        unit_data = self.node_data[unit_id]
        unit_tags = set(unit_data.get("tags", []))
        unit_type = unit_data.get("type", "unknown")

        related_units = []

        for other_id, other_data in self.node_data.items():
            if other_id == unit_id:
                continue

            other_tags = set(other_data.get("tags", []))
            other_type = other_data.get("type", "unknown")

            # Simple similarity based on shared tags and type
            if other_type == unit_type and unit_tags & other_tags:
                similarity_score = len(unit_tags & other_tags)
                related_units.append({
                    "unit": other_data,
                    "relationship": {"type": "similar", "weight": similarity_score},
                    "depth": 1
                })

                if len(related_units) >= limit:
                    break

        return related_units

    def get_metadata(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific unit."""
        unit = self.retrieve(unit_id)
        if unit and "metadata" in unit:
            return unit["metadata"]
        return None

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph structure."""
        if self.driver:
            return self._get_neo4j_stats()
        else:
            return self._get_networkx_stats()

    def _get_neo4j_stats(self) -> Dict[str, Any]:
        """Get Neo4j graph statistics."""
        with self.driver.session() as session:
            # Node count
            node_result = session.run("MATCH (n:Memory) RETURN count(n) as nodes")
            node_count = node_result.single()["nodes"]
            node_result.consume()

            # Relationship count and types
            rel_result = session.run("MATCH ()-[r]-() RETURN type(r) as type, count(r) as count")
            relationships = {}
            for record in rel_result:
                relationships[record["type"]] = record["count"]
            rel_result.consume()

        return {
            "node_count": node_count,
            "relationship_types": relationships,
            "storage_type": "neo4j"
        }


    def _get_networkx_stats(self) -> Dict[str, Any]:
        """Get NetworkX graph statistics or fallback stats."""
        if not self.graph:
            return {
                "nodes": len(self.node_data),
                "relationships": {"total": 0, "by_type": {}},
                "storage_type": "dict_fallback"
            }

        return {
            "nodes": self.graph.number_of_nodes(),
            "relationships": {
                "total": self.graph.number_of_edges(),
                "by_type": {}  # Could be extended to count edge types
            },
            "storage_type": "networkx"
        }