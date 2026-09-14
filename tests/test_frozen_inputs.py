"""Frozen transport failures must not relabel partial bytes as valid data."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
from research import hosted_study


class FrozenInputTests(unittest.TestCase):
    payload=b'fixed-input-part'

    def call(self, directory):
        self.assertTrue(hasattr(hosted_study,'get_part'),'missing bounded frozen-part retrieval')
        with patch.object(hosted_study,'PARTS',(hashlib.sha256(self.payload).hexdigest(),)):
            return hosted_study.get_part(Path(directory),0)

    def test_transport_reset_retries_without_changing_identity(self):
        with tempfile.TemporaryDirectory() as d, patch.object(hosted_study.urllib.request,'urlopen',
                side_effect=[URLError(ConnectionResetError('reset')),io.BytesIO(self.payload)]) as request, \
                patch('time.sleep'):
            result=self.call(d)
            self.assertEqual(result.read_bytes(),self.payload)
            self.assertEqual(request.call_count,2)
            self.assertFalse(list(Path(d).glob('*.partial')))

    def test_verified_existing_part_is_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as d, patch.object(hosted_study.urllib.request,'urlopen') as request:
            (Path(d)/'evidence.tar.gz.part-000').write_bytes(self.payload)
            self.assertEqual(self.call(d).read_bytes(),self.payload)
            request.assert_not_called()

    def test_partial_or_corrupt_download_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as d, patch.object(hosted_study.urllib.request,'urlopen',
                side_effect=lambda *a,**k:io.BytesIO(b'corrupt')) as request, patch('time.sleep'):
            with self.assertRaises(ValueError):self.call(d)
            self.assertEqual(request.call_count,3)
            self.assertFalse((Path(d)/'evidence.tar.gz.part-000').exists())
            self.assertFalse(list(Path(d).glob('*.partial')))

    def test_failed_transport_has_bounded_attempts(self):
        with tempfile.TemporaryDirectory() as d, patch.object(hosted_study.urllib.request,'urlopen',
                side_effect=URLError('offline')) as request, patch('time.sleep'):
            with self.assertRaises(URLError):self.call(d)
            self.assertEqual(request.call_count,3)
            self.assertFalse(list(Path(d).glob('*.partial')))


if __name__=='__main__':unittest.main()
